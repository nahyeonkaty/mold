import glob
import os
import random
from copy import deepcopy
from io import BytesIO

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from numpy.typing import NDArray
from PIL import Image
from scipy.ndimage.filters import gaussian_filter
from sklearn.metrics import accuracy_score, average_precision_score
from torch.utils.data import DataLoader, Dataset


SEED: int = 0

MEAN: dict[str, list[float]] = {
    "imagenet": [0.485, 0.456, 0.406],
    "clip": [0.48145466, 0.4578275, 0.40821073],
}

STD: dict[str, list[float]] = {
    "imagenet": [0.229, 0.224, 0.225],
    "clip": [0.26862954, 0.26130258, 0.27577711],
}


def find_best_threshold(
    y_true: NDArray[np.floating], y_pred: NDArray[np.floating]
) -> float:
    """Find the best threshold for binary classification.

    We assume first half is real (0), and the second half is fake (1).
    """
    N = y_true.shape[0]

    if y_pred[0 : N // 2].max() <= y_pred[N // 2 : N].min():  # perfectly separable case
        return float((y_pred[0 : N // 2].max() + y_pred[N // 2 : N].min()) / 2)

    best_acc: float = 0
    best_thres: float = 0
    for thres in y_pred:
        temp = deepcopy(y_pred)
        temp[temp >= thres] = 1
        temp[temp < thres] = 0

        acc = (temp == y_true).sum() / N
        if acc >= best_acc:
            best_thres = float(thres)
            best_acc = acc

    return best_thres


def png2jpg(img: Image.Image, quality: int) -> Image.Image:
    """Convert PNG image to JPEG with specified quality."""
    out = BytesIO()
    img.save(out, format="jpeg", quality=quality)  # ranging from 0-95, 75 is default
    img = Image.open(out)
    # load from memory before ByteIO closes
    img = np.array(img)
    out.close()
    return Image.fromarray(img)


def gaussian_blur(img: Image.Image, sigma: float) -> Image.Image:
    """Apply Gaussian blur to an image."""
    img_arr = np.array(img)

    gaussian_filter(img_arr[:, :, 0], output=img_arr[:, :, 0], sigma=sigma)
    gaussian_filter(img_arr[:, :, 1], output=img_arr[:, :, 1], sigma=sigma)
    gaussian_filter(img_arr[:, :, 2], output=img_arr[:, :, 2], sigma=sigma)

    return Image.fromarray(img_arr)


def calculate_acc(
    y_true: NDArray[np.floating], y_pred: NDArray[np.floating], thres: float
) -> tuple[float, float, float]:
    """Calculate accuracy metrics.

    Returns:
        Tuple of (real_accuracy, fake_accuracy, overall_accuracy).
    """
    r_acc = accuracy_score(y_true[y_true == 0], y_pred[y_true == 0] > thres)
    f_acc = accuracy_score(y_true[y_true == 1], y_pred[y_true == 1] > thres)
    acc = accuracy_score(y_true, y_pred > thres)
    return float(r_acc), float(f_acc), float(acc)


@torch.no_grad()
def validate(
    model: nn.Module, loader: DataLoader, find_thres: bool = False
) -> (
    tuple[float, float, float, float]
    | tuple[float, float, float, float, float, float, float, float]
):
    """Validate the model on a dataset.

    Args:
        model: The model to validate.
        loader: DataLoader for the validation dataset.
        find_thres: Whether to find the best threshold.

    Returns:
        If find_thres is False: (ap, r_acc, f_acc, acc)
        If find_thres is True: (ap, r_acc0, f_acc0, acc0, r_acc1, f_acc1, acc1, best_thres)
    """
    y_true: list[float] = []
    y_pred: list[float] = []
    print(f"Length of dataset: {len(loader)}")
    for img, label in loader:
        in_tens = img.cuda()

        y_pred.extend(model(in_tens).sigmoid().flatten().tolist())
        y_true.extend(label.flatten().tolist())

        print(f"validating... {len(y_pred)} samples done")

    y_true_arr, y_pred_arr = np.array(y_true), np.array(y_pred)

    # if wrong, save the images
    # save_dir = "wrong-glidefromglide-learnable-filename"
    # print("=======================================================")
    # print(f"Saving misclassified images to {save_dir}")
    # print("=======================================================")
    # os.makedirs(save_dir, exist_ok=True)

    # with torch.no_grad():
    #     y_true, y_pred = [], []
    #     misclassified_indices = []
    #     misclassified_images = []
    #     filenames = []

    #     print("Length of dataset: %d" % (len(loader)))
    #     for idx, (img, label, filename) in enumerate(loader):
    #         in_tens = img.cuda()
    #         preds = model(in_tens).sigmoid().flatten()

    #         y_pred.extend(preds.tolist())
    #         y_true.extend(label.flatten().tolist())
    #         filenames.extend(filename)

    #         # Find misclassified samples
    #         for i in range(len(label)):
    #             pred_label = 1 if preds[i] >= 0.5 else 0
    #             true_label = int(label[i].item())
    #             if pred_label != true_label:
    #                 misclassified_indices.append(len(y_true) - len(label) + i)
    #                 misclassified_images.append(img[i])

    #         print(f"Validating... {len(y_pred)} samples done")

    #     # Convert lists to numpy arrays
    #     y_true, y_pred = np.array(y_true), np.array(y_pred)

    #     # Save misclassified images
    #     for i, img_tensor in zip(misclassified_indices, misclassified_images):
    #         # normalize the image
    #         img_tensor = (img_tensor - img_tensor.min()) / (img_tensor.max() - img_tensor.min())
    #         img_pil = Image.fromarray((img_tensor.numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
    #         truth = "real" if y_true[i] == 0 else "fake"
    #         img_pil.save(os.path.join(save_dir, f"{truth}_{y_pred[i]}_{filenames[i]}.png"))

    #     print(f"Saved {len(misclassified_images)} misclassified images.")

    # ================== save this if you want to plot the curves =========== #
    # torch.save( torch.stack( [torch.tensor(y_true), torch.tensor(y_pred)] ),  'baseline_predication_for_pr_roc_curve.pth' )
    # exit()
    # =================================================================== #

    # Compute average precision score (AP).
    ap = average_precision_score(y_true_arr, y_pred_arr)

    # Acc based on 0.5.
    r_acc0, f_acc0, acc0 = calculate_acc(y_true_arr, y_pred_arr, 0.5)
    if not find_thres:
        return float(ap), r_acc0, f_acc0, acc0

    # Acc based on the best thres
    best_thres = find_best_threshold(y_true_arr, y_pred_arr)
    r_acc1, f_acc1, acc1 = calculate_acc(y_true_arr, y_pred_arr, best_thres)

    return float(ap), r_acc0, f_acc0, acc0, r_acc1, f_acc1, acc1, best_thres


# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = #


def recursively_read(
    rootdir: str,
    must_contain: str,
    exts: list[str] = ["png", "jpg", "JPEG", "jpeg", "bmp"],
) -> list[str]:
    """Recursively read image files from a directory."""
    out: list[str] = []
    for r, d, f in os.walk(rootdir):
        for file in f:
            if (file.split(".")[-1] in exts) and (
                must_contain in os.path.join(r, file)
            ):
                out.append(os.path.join(r, file))
    return out


def get_list(path: str, must_contain: str = "") -> list[str]:
    """Get list of image files from a path."""
    image_list = glob.glob(path + "/*")
    return image_list


class RealFakeDataset(Dataset):
    """Dataset for real and fake image classification."""

    def __init__(
        self,
        real_path: str | list[str],
        fake_path: str | list[str],
        data_mode: str,
        max_sample: int,
        arch: str,
        jpeg_quality: int | None = None,
        gaussian_sigma: float | None = None,
        cropsize: int = 224,
        resize_scale: float | None = None,
    ) -> None:
        assert data_mode in ["wang2020", "ours", "custom"]
        self.jpeg_quality = jpeg_quality
        self.gaussian_sigma = gaussian_sigma
        self.resize_scale = resize_scale

        # Get image paths.
        if isinstance(real_path, str) and isinstance(fake_path, str):
            real_list, fake_list = self.read_path(
                real_path, fake_path, data_mode, max_sample
            )
        else:
            real_list: list[str] = []
            fake_list: list[str] = []
            for real_p, fake_p in zip(real_path, fake_path):
                real_l, fake_l = self.read_path(real_p, fake_p, data_mode, max_sample)
                real_list += real_l
                fake_list += fake_l

        self.total_list = real_list + fake_list

        # Set labels.
        self.labels_dict: dict[str, int] = {}
        for i in real_list:
            self.labels_dict[i] = 0
        for i in fake_list:
            self.labels_dict[i] = 1

        stat_from = "imagenet" if arch.lower().startswith("imagenet") else "clip"
        self.transform = transforms.Compose(
            [
                transforms.CenterCrop(cropsize),
                transforms.ToTensor(),
                transforms.Normalize(mean=MEAN[stat_from], std=STD[stat_from]),
            ]
        )

    def read_path(
        self, real_path: str, fake_path: str, data_mode: str, max_sample: int
    ) -> tuple[list[str], list[str]]:
        """Read and balance image paths."""
        if data_mode == "wang2020":
            real_list = get_list(real_path, must_contain="0_real")
            fake_list = get_list(fake_path, must_contain="1_fake")
        else:
            real_list = get_list(real_path)
            fake_list = get_list(fake_path)

        max_sample = min(len(real_list), len(fake_list))
        print("max_sample:", max_sample)
        random.shuffle(real_list)
        random.shuffle(fake_list)
        real_list = real_list[0:max_sample]
        fake_list = fake_list[0:max_sample]

        assert len(real_list) == len(fake_list)

        return real_list, fake_list

    def __len__(self) -> int:
        return len(self.total_list)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, str]:
        img_path = self.total_list[idx]

        label = self.labels_dict[img_path]
        img = Image.open(img_path).convert("RGB")

        if self.gaussian_sigma is not None:
            img = gaussian_blur(img, self.gaussian_sigma)
        if self.jpeg_quality is not None:
            img = png2jpg(img, self.jpeg_quality)
        if self.resize_scale is not None:
            h, w = img.size
            img = img.resize((int(h * self.resize_scale), int(w * self.resize_scale)))

        img_tensor = self.transform(img)
        return img_tensor, label, img_path.split("/")[-1]
