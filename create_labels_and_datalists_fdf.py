import argparse
from collections import defaultdict
import json
import os
import os.path as osp
import cv2
import numpy as np
from tqdm import tqdm
import imagesize


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=str,
        help="Path to data root.",
        default="/home/s0001396/Documents/phd/datasets/fdf256"
    )
    parser.add_argument(
        "--output-folder",
        type=str,
        help="Path to output folder.",
        default="/home/s0001396/Documents/phd/wasp/as/yolov7/coco"
    )
    return parser.parse_args()


def get_image_filenames(path):
    files =  os.listdir(osp.join(path, "images"))
    # filter out all the files that are not images jpg or png
    files = [f for f in files if f.endswith(".jpg") or f.endswith(".png")]
    return files

def load_annotations(path):
    return np.load(osp.join(path, "bounding_box.npy"))

def get_image_sizes(path, files):
    sizes = []
    for f in tqdm(files):
        sizes.append(imagesize.get(osp.join(path, f)))
    return np.array(sizes)

def write_annofiles(folder, output_folder):
    split = folder.split("/")[-1]
    files = get_image_filenames(folder)
    sizes = get_image_sizes(osp.join(folder, "images"), files).astype(np.float32)
    image_index = [int(f.split(".")[0]) for f in files]

    bounding_boxes = load_annotations(folder).astype(np.float32)

    # only keep bounding boxes in images_index
    bounding_boxes = bounding_boxes[image_index]

    # normalize bounding boxes
    # sizes is a numpy array of shape (N, 2) where N is the number of images
    # sizes[:, 0] is the width of the image
    # sizes[:, 1] is the height of the image
    bounding_boxes[:, 0] /= sizes[:, 0]
    bounding_boxes[:, 1] /= sizes[:, 1]
    bounding_boxes[:, 2] /= sizes[:, 0]
    bounding_boxes[:, 3] /= sizes[:, 1]


    # bounding boxes are in the format x1, y1, x2, y2
    # go from x1, y1, x2, y2 to x1, y1, x2, y1, x2, y2, x1, y2
    x1 = bounding_boxes[:, 0]
    y1 = bounding_boxes[:, 1]
    x2 = bounding_boxes[:, 2]
    y2 = bounding_boxes[:, 3]
    bounding_boxes = np.stack([x1, y1, x2, y1, x2, y2, x1, y2], axis=1)


    # prepend 0 to the bounding boxe coordinates to indicate first class
    # separate bounding boxes with a space
    # save to file output_folder/labels/fdf/{split}/{image_index}.txt
    for i, b in tqdm(zip(image_index, bounding_boxes)):
        with open(osp.join(output_folder, "labels/fdf/{}/{}.txt".format(split, i)), "w") as file:
            file.write("0 {} {} {} {} {} {} {} {}\n".format(*b))

    # write images filepaths as ./images/fdf/{split}/image_filename
    with open(osp.join(output_folder, "{}.txt".format(split)), "w") as file:
        for i, idx in enumerate(image_index):
            file.write(f"./images/fdf/{split}/{files[i]}\n")

def main():

    args = _parse_args()

    data_root = args.data_root
    train_root = osp.join(data_root, "train")
    val_root = osp.join(data_root, "val")

    write_annofiles(train_root, args.output_folder)
    write_annofiles(val_root, args.output_folder)

    

if __name__ == "__main__":
    main()