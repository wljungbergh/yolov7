import argparse
from collections import defaultdict
import json
from pycocotools.mask import decode as decode_mask
import cv2
import numpy as np
from tqdm import tqdm

HEAD_IDX = 13
BAD_LICENSES = [0,1,2]


def get_densepose_mask(polygons):
    """Extract the mask of the densepose annotation, note that it is only extracted for the head."""
    generated_mask = np.zeros([256, 256])
    if polygons[HEAD_IDX]:
        current_mask = decode_mask(polygons[HEAD_IDX])
        generated_mask[current_mask > 0] = 1
    return generated_mask


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        type=str,
        help="Path to the annotations file that is downloaded from the coco website.",
    )
    parser.add_argument("--train", action="store_true")
    return parser.parse_args()


def main():

    args = _parse_args()
    train = args.train
    input_file = args.input_file

    train_or_val = "train" if train else "val"

    # load the annotations file
    with open(input_file, "r") as f:
        data = json.load(f)

    
    images = [img for img in data["images"] if int(img["license"]) not in BAD_LICENSES]
    img_ids = set([img["id"] for img in images])
    annotations = [ann for ann in data["annotations"] if ann["image_id"] in img_ids]

    # get the height and width of the each image
    image_id_to_hw = {
        image["id"]: (image["height"], image["width"]) for image in images
    }

    image_id_to_bboxes = defaultdict(list)
    for ann in tqdm(annotations):
        # ignore all annotations that are not densepose
        if not "dp_masks" in ann:
            continue

        h, w = image_id_to_hw[ann["image_id"]]
        bbr = np.array(ann["bbox"]).astype(int)  # the box.
        mask = get_densepose_mask(ann["dp_masks"])

        x1, y1, x2, y2 = bbr[0], bbr[1], bbr[0] + bbr[2], bbr[1] + bbr[3]
        x2 = min([x2, w])
        y2 = min([y2, h])

        # resize the mask to the bounding box size
        masked_img = cv2.resize(
            mask, (int(x2 - x1), int(y2 - y1)), interpolation=cv2.INTER_NEAREST
        )

        idxes = np.nonzero(masked_img)
        # if we dont have any faces we ignore this annotation
        if len(idxes[0]) == 0:
            continue

        # extract the bounding box
        ymin = (np.min(idxes[0]) + y1) / h
        ymax = (np.max(idxes[0]) + y1) / h
        xmin = (np.min(idxes[1]) + x1) / w
        xmax = (np.max(idxes[1]) + x1) / w

        image_id_to_bboxes[ann["image_id"]].append(
            [
                xmin,
                ymin,
                xmax,
                ymin,
                xmax,
                ymax,
                xmin,
                ymax,
            ]
        )

    # write all the annotations into the correct format
    for image_id, labels in image_id_to_bboxes.items():
        filename = str(image_id).zfill(12)

        label_filename = f"COCO_{train_or_val}2014_{filename}.txt"

        polystring = ""
        for label in labels:
            polystring += "0 "
            polystring += " ".join([str(x) for x in label]) + "\n"

        with open(f"coco/labels/{train_or_val}2014/" + label_filename, "w") as f:
            f.write(polystring)

    # write the train.txt and val.txt files containig the paths to the images
    filenames = [
        f"./images/{train_or_val}2014/" + image["file_name"] for image in images
    ]
    with open(f"coco/{train_or_val}2014.txt", "a") as f:
        f.write("\n".join(filenames))

if __name__ == "__main__":
    main()