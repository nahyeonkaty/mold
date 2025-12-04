from __future__ import annotations

import glob
import os
from argparse import Namespace
from io import BytesIO
from random import choice, random, shuffle

import cv2
import numpy as np
import torchvision.transforms.functional as TF
from PIL import Image, ImageFile
from scipy.ndimage.filters import gaussian_filter
from torch.utils.data import Dataset
from torchvision.transforms import v2 as transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True


MEAN: dict[str, list[float]] = {
    "imagenet": [0.485, 0.456, 0.406],
    "clip": [0.48145466, 0.4578275, 0.40821073],
}

STD: dict[str, list[float]] = {
    "imagenet": [0.229, 0.224, 0.225],
    "clip": [0.26862954, 0.26130258, 0.27577711],
}


def recursively_read(
    rootdir: str, must_contain: str, exts: list[str] = ["png", "jpg", "JPEG", "jpeg"]
) -> list[str]:
    """Recursively read image files from a directory."""
    out: list[str] = []
    for r, d, f in os.walk(rootdir):
        for file in f:
            if (file.split(".")[1] in exts) and (must_contain in os.path.join(r, file)):
                out.append(os.path.join(r, file))
    return out


def get_list(path: str, mode: str, must_contain: str = "", gm: str = "") -> list[str]:
    """Get list of image files from a path."""
    if "genimage" in path:
        image_list = glob.glob(path + f"/{gm}/{mode}/{must_contain}/*")
    elif "progan" in path:
        image_list = glob.glob(path + f"/{mode}/{gm}/*/{must_contain}/*")
    else:
        raise ValueError("Invalid path")
    return image_list


def create_transform(opt: Namespace) -> transforms.Compose | None:
    """Create image transformation pipeline."""
    if opt.is_train:
        crop_func = transforms.RandomCrop(opt.cropSize)
    elif opt.no_crop:
        crop_func = transforms.Lambda(lambda img: img)
    else:
        crop_func = transforms.CenterCrop(opt.cropSize)

    if opt.is_train and not opt.no_flip:
        flip_func = transforms.RandomHorizontalFlip()
    else:
        flip_func = transforms.Lambda(lambda img: img)
    if not opt.is_train and opt.no_resize:
        rz_func = transforms.Lambda(lambda img: img)
    else:
        rz_func = transforms.Lambda(lambda img: custom_resize(img, opt))

    if "clip" in opt.arch.lower():
        stat_from = "clip"
    else:
        stat_from = "imagenet"  # ImageNet default stats.

    print("mean and std stats are from: ", stat_from)
    if "2b" not in opt.arch:
        print("using Official CLIP's normalization")
        transform = transforms.Compose(
            [
                rz_func,
                transforms.Lambda(lambda img: data_augment(img, opt)),
                crop_func,
                flip_func,
                transforms.ToTensor(),
                transforms.Normalize(mean=MEAN[stat_from], std=STD[stat_from]),
            ]
        )
    else:
        print("Using CLIP 2B transform")
        transform = None  # will be initialized in trainer.py
    return transform


class RealFakeDataset(Dataset):
    """Dataset for real and fake image classification."""

    def __init__(self, opt: Namespace) -> None:
        assert opt.data_label in ["train", "val"]
        self.data_label: str = opt.data_label

        real_list: list[str]
        fake_list: list[str]

        if opt.data_mode == "ours":
            pickle_name = "train.pickle" if opt.data_label == "train" else "val.pickle"
            real_list = get_list(os.path.join(opt.real_list_path, pickle_name))
            fake_list = get_list(os.path.join(opt.fake_list_path, pickle_name))
        elif opt.data_mode == "wang2020":
            temp = "train/progan" if opt.data_label == "train" else "test/progan"
            real_list = get_list(
                os.path.join(opt.wang2020_data_path, temp), must_contain="0_real"
            )
            fake_list = get_list(
                os.path.join(opt.wang2020_data_path, temp), must_contain="1_fake"
            )
        elif opt.data_mode == "ours_wang2020":
            pickle_name = "train.pickle" if opt.data_label == "train" else "val.pickle"
            real_list = get_list(os.path.join(opt.real_list_path, pickle_name))
            fake_list = get_list(os.path.join(opt.fake_list_path, pickle_name))
            temp = "train/progan" if opt.data_label == "train" else "test/progan"
            real_list += get_list(
                os.path.join(opt.wang2020_data_path, temp), must_contain="0_real"
            )
            fake_list += get_list(
                os.path.join(opt.wang2020_data_path, temp), must_contain="1_fake"
            )
        elif opt.data_mode == "custom":
            mode = "train" if opt.data_label == "train" else "val"
            data_dir = opt.data_root
            print("==========================")
            print("data_dir: ", data_dir)
            print("opt.gm: ", opt.gm)
            print(f"opt.data_label: {opt.data_label}")
            print(f"opt.data_mode: {opt.data_mode}")
            print("==========================")
            real_list = get_list(data_dir, mode, must_contain="real", gm=opt.gm)
            fake_list = get_list(data_dir, mode, must_contain="fake", gm=opt.gm)
        elif opt.data_mode == "progan_custom":
            mode = "train" if opt.data_label == "train" else "val"
            data_dir = opt.data_root
            print("==========================")
            print("data_dir: ", data_dir)
            print("opt.gm: ", opt.gm)
            print("==========================")
            real_list = get_list(data_dir, mode, must_contain="real", gm=opt.gm)
            fake_list = get_list(data_dir, mode, must_contain="fake", gm=opt.gm)
        else:
            raise ValueError(f"Unknown data_mode: {opt.data_mode}")

        # Setting the labels for the dataset.
        self.labels_dict: dict[str, int] = {}
        for i in real_list:
            self.labels_dict[i] = 0
        for i in fake_list:
            self.labels_dict[i] = 1

        self.total_list: list[str] = real_list + fake_list
        shuffle(self.total_list)
        self.transform = create_transform(opt)

    def __len__(self) -> int:
        return len(self.total_list)

    def __getitem__(self, idx: int) -> tuple:
        img_path = self.total_list[idx]
        label = self.labels_dict[img_path]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return img, label


def data_augment(img: Image.Image, opt: Namespace) -> Image.Image:
    """Apply data augmentation to an image."""
    img_arr = np.array(img)
    if img_arr.ndim == 2:
        img_arr = np.expand_dims(img_arr, axis=2)
        img_arr = np.repeat(img_arr, 3, axis=2)

    if random() < opt.blur_prob:
        sig = sample_continuous(opt.blur_sig)
        gaussian_blur(img_arr, sig)

    if random() < opt.jpg_prob:
        method = sample_discrete(opt.jpg_method)
        qual = sample_discrete(opt.jpg_qual)
        img_arr = jpeg_from_key(img_arr, qual, method)

    return Image.fromarray(img_arr)


def sample_continuous(s: list[float]) -> float:
    """Sample a continuous value from a range."""
    if len(s) == 1:
        return s[0]
    if len(s) == 2:
        rg = s[1] - s[0]
        return random() * rg + s[0]
    raise ValueError("Length of iterable s should be 1 or 2.")


def sample_discrete(s: list) -> any:
    """Sample a discrete value from a list."""
    if len(s) == 1:
        return s[0]
    return choice(s)


def gaussian_blur(img: np.ndarray, sigma: float) -> None:
    """Apply Gaussian blur in-place."""
    gaussian_filter(img[:, :, 0], output=img[:, :, 0], sigma=sigma)
    gaussian_filter(img[:, :, 1], output=img[:, :, 1], sigma=sigma)
    gaussian_filter(img[:, :, 2], output=img[:, :, 2], sigma=sigma)


def cv2_jpg(img: np.ndarray, compress_val: int) -> np.ndarray:
    """Compress image using OpenCV JPEG compression."""
    img_cv2 = img[:, :, ::-1]
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_val]
    result, encimg = cv2.imencode(".jpg", img_cv2, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg[:, :, ::-1]


def pil_jpg(img: np.ndarray, compress_val: int) -> np.ndarray:
    """Compress image using PIL JPEG compression."""
    out = BytesIO()
    pil_img = Image.fromarray(img)
    pil_img.save(out, format="jpeg", quality=compress_val)
    pil_img = Image.open(out)
    # load from memory before ByteIO closes
    result = np.array(pil_img)
    out.close()
    return result


jpeg_dict: dict[str, callable] = {"cv2": cv2_jpg, "pil": pil_jpg}


def jpeg_from_key(img: np.ndarray, compress_val: int, key: str) -> np.ndarray:
    """Apply JPEG compression using the specified method."""
    method = jpeg_dict[key]
    return method(img, compress_val)


rz_dict: dict[str, int] = {
    "bilinear": Image.BILINEAR,
    "bicubic": Image.BICUBIC,
    "lanczos": Image.LANCZOS,
    "nearest": Image.NEAREST,
}


def custom_resize(img: Image.Image, opt: Namespace) -> Image.Image:
    """Resize image with random interpolation method."""
    interp = sample_discrete(opt.rz_interp)
    return TF.resize(img, opt.loadSize, interpolation=rz_dict[interp])
