import argparse
import torch
import yaml
from tqdm import tqdm
import os

from models.yolo_encoder import Encoder
from utils.datasets import create_dataloader
from utils.general import (
    check_dataset,
    check_file,
    check_img_size,
    colorstr,
)
from utils.torch_utils import select_device


def main(data, batch_size=32, imgsz=640, half_precision=True):
    device = select_device(opt.device, batch_size=batch_size)

    enc = Encoder(opt.cfg).to(device)

    gs = max(int(enc.stride.max()), 32)  # grid size (max stride)
    imgsz = check_img_size(imgsz, s=gs)  # check img_size

    # Half
    half = (
        device.type != "cpu" and half_precision
    )  # half precision only supported on CUDA
    if half:
        enc.half()

    enc.eval()
    if isinstance(data, str):
        with open(data) as f:
            data = yaml.load(f, Loader=yaml.SafeLoader)
    check_dataset(data)  # check

    # Dataloader
    if device.type != "cpu":
        enc(
            torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(enc.parameters()))
        )  # run once
    task = opt.task

    dataloader = create_dataloader(
        data[task],
        imgsz,
        batch_size,
        gs,
        opt,
        pad=0.5,
        rect=True,
        prefix=colorstr(f"{task}: "),
    )[0]

    for img, _, paths, _ in tqdm(dataloader, desc=f"Caching features for {task}"):
        img = img.to(device, non_blocking=True)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        with torch.no_grad():
            # Run model
            out = enc(img, augment=False)  # inference and training outputs

        for i, path in enumerate(paths):
            filename = os.path.basename(path).split("_")[0]
            torch.save(
                out[i],
                f"{opt.out_folder}/{task}/{filename}.pt",
            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="cache_features.py")
    parser.add_argument("--cfg", type=str, default=None, help="config path")
    parser.add_argument(
        "--data", type=str, default="data/coco.yaml", help="*.data path"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="size of each image batch"
    )
    parser.add_argument(
        "--img-size", type=int, default=640, help="inference size (pixels)"
    )
    parser.add_argument(
        "--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu"
    )
    parser.add_argument(
        "--single-cls", action="store_true", help="treat as single-class dataset"
    )
    parser.add_argument(
        "--out-folder",
        type=str,
        help="path to the directory where cached features should be stored",
        required=True,
    )
    parser.add_argument(
        "--task",
        type=str,
        help="task for which features should be cached",
        required=True,
    )
    opt = parser.parse_args()
    opt.data = check_file(opt.data)  # check file
    print(opt)
    main(opt.data, opt.batch_size, opt.img_size)
