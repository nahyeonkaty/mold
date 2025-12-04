#!/usr/bin/env python3
import argparse
import os

import torch
from torch.utils.data import DataLoader

from mold.data.dataset_paths import DATASET_PATHS
from mold.detector import Detector
from mold.utils import set_seed
from mold.validation import RealFakeDataset, validate


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--real_path", type=str, default=None, help="dir name or a pickle"
    )
    parser.add_argument(
        "--fake_path", type=str, default=None, help="dir name or a pickle"
    )
    parser.add_argument("--data_mode", type=str, default=None, help="wang2020 or ours")
    parser.add_argument(
        "--max_sample",
        type=int,
        default=1000,
        help="only check this number of images for both fake/real",
    )

    parser.add_argument("--arch", type=str, default="res50")
    parser.add_argument(
        "--ckpt", type=str, default="./pretrained_weights/fc_weights.pth"
    )

    parser.add_argument("--result_folder", type=str, default="result", help="")
    parser.add_argument("--batch_size", type=int, default=128)

    parser.add_argument(
        "--jpeg_quality",
        type=int,
        default=None,
        help="100, 90, 80, ... 30. Used to test robustness of our model. Not apply if None",
    )
    parser.add_argument(
        "--gaussian_sigma",
        type=int,
        default=None,
        help="0,1,2,3,4.     Used to test robustness of our model. Not apply if None",
    )
    parser.add_argument(
        "--resize_scale",
        type=float,
        default=None,
        help="0.5, 1.0, 1.5, ... Used to test robustness of our model. Not apply if None",
    )

    parser.add_argument(
        "--truncate_layer",
        type=int,
        default=3,
        help="where to truncate the network (layer idx)",
    )
    parser.add_argument("--crop_size", type=int, default=224, help="crop size")
    parser.add_argument(
        "--skip_idx", default=None, type=int, help="skip the transformer layer"
    )

    parser.add_argument(
        "--seed", type=int, default=0, help="random seed for reproducibility"
    )
    opt = parser.parse_args()

    # Prepare the model.
    model = Detector(opt.arch)
    print("opt.ckpt:", opt.ckpt)
    state_dict = torch.load(opt.ckpt, map_location="cpu")
    model.load_state_dict(state_dict["model"])
    print("Model loaded..")
    model.eval().cuda()

    # Prepare datasets.
    if (opt.real_path == None) or (opt.fake_path == None) or (opt.data_mode == None):
        dataset_paths = DATASET_PATHS
    else:
        dataset_paths = [
            dict(
                real_path=opt.real_path,
                fake_path=opt.fake_path,
                data_mode=opt.data_mode,
            )
        ]

    if isinstance(opt.result_folder, list):
        for folder in opt.result_folder:
            os.makedirs(folder, exist_ok=True)
    else:
        os.makedirs(opt.result_folder, exist_ok=True)

    for dataset_path in dataset_paths:
        set_seed(opt.seed)
        print(f"real_path: {dataset_path['real_path']}")
        print(f"fake_path: {dataset_path['fake_path']}")
        print(f"data_mode: {dataset_path['data_mode']}")
        dataset = RealFakeDataset(
            dataset_path["real_path"],
            dataset_path["fake_path"],
            dataset_path["data_mode"],
            opt.max_sample,
            opt.arch,
            jpeg_quality=opt.jpeg_quality,
            gaussian_sigma=opt.gaussian_sigma,
            cropsize=opt.crop_size,
            resize_scale=opt.resize_scale,
        )

        loader = DataLoader(
            dataset,
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=4,
        )

        # ap, r_acc0, f_acc0, acc0, r_acc1, f_acc1, acc1, best_thres = validate(model, loader, find_thres=True)
        # print('ap: ', str(round(ap*100, 2))+'\n')
        # print('racc: ', str(round(r_acc0*100, 2))+' facc: '+str(round(f_acc0*100, 2))+' acc0: '+str(round(acc0*100, 2))+'\n' )

        ap, r_acc0, f_acc0, acc0 = validate(model, loader, find_thres=False)
        print("ap: ", str(round(ap * 100, 2)) + "\n")
        print(
            "racc: ",
            str(round(r_acc0 * 100, 2))
            + " facc: "
            + str(round(f_acc0 * 100, 2))
            + " acc0: "
            + str(round(acc0 * 100, 2))
            + "\n",
        )


if __name__ == "__main__":
    main()
