from itertools import repeat
import os
from typing import List
import typer
from tqdm.contrib.concurrent import process_map

from zod import ZodFrames
from zod.data_classes.frame import ZodFrame
from zod.anno.object import AnnotatedObject
from zod.constants import TRAIN, VAL, Anonymization, AnnotationProject

FILENAME = "zod_{version}_{anonymization}_{train_or_val}.txt"

ZOD_CLASS_MAP = {
    "Vehicle": 0,
    "VulnerableVehicle": 1,
    "Pedestrian": 2,
}
IMAGE_SIZE = (3848, 2168)


def _get_label_strs(frame: ZodFrame) -> List[str]:
    objs: List[AnnotatedObject] = frame.get_annotation(
        AnnotationProject.OBJECT_DETECTION
    )
    label_strs = []
    for obj in objs:
        if obj.name not in ZOD_CLASS_MAP or obj.unclear:
            continue
        # the yolov7 framework can not handle coordinates outside of the image
        obj.box2d.xyxy[0] = max(obj.box2d.xyxy[0], 0)
        obj.box2d.xyxy[1] = max(obj.box2d.xyxy[1], 0)
        obj.box2d.xyxy[2] = min(obj.box2d.xyxy[2], IMAGE_SIZE[0])
        obj.box2d.xyxy[3] = min(obj.box2d.xyxy[3], IMAGE_SIZE[1])

        coords = [
            obj.box2d.center[0] / IMAGE_SIZE[0],
            obj.box2d.center[1] / IMAGE_SIZE[1],
            obj.box2d.dimension[0] / IMAGE_SIZE[0],
            obj.box2d.dimension[1] / IMAGE_SIZE[1],
        ]
        label_str = (
            str(ZOD_CLASS_MAP[obj.name]) + " " + " ".join([str(c) for c in coords])
        )
        label_strs.append(label_str)
    return label_strs


def process_frame(
    frame: ZodFrame,
    anonymization: Anonymization,
    output_dir: str,
    train_frames: List[str],
    val_frames: List[str],
):
    camera_frame = frame.get_camera_frame(Anonymization(anonymization))
    # get the label file path
    label_path = os.path.join(
        output_dir,
        "labels",
        os.path.basename(camera_frame.filepath).replace(".jpg", ".txt"),
    )
    # we might have already written the label file if we've run the script
    # before with a different anonymization method
    if True or not os.path.exists(label_path):
        label_strs = _get_label_strs(frame)
        with open(label_path, "w") as f:
            f.write("\n".join(label_strs))
    if frame.info.id in train_frames:
        return camera_frame.filepath, None
    elif frame.info.id in val_frames:
        return None, camera_frame.filepath


def main(
    zod_root: str = typer.Option(
        ..., help="Path to the root directory of the ZOD dataset"
    ),
    output_dir: str = typer.Option(..., help="Path to the output directory"),
    mini: bool = typer.Option(False, help="Use the mini version of the dataset"),
    anonymization: str = typer.Option("blur", help="Anonymization method to use"),
):
    version = "mini" if mini else "full"
    assert anonymization in [v.value for v in Anonymization]
    zod_frames = ZodFrames(zod_root, version=version)

    # create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels"), exist_ok=True)

    # get the train/val split
    train_frames = zod_frames.get_split(TRAIN)
    val_frames = zod_frames.get_split(VAL)

    # process the frames in parallel
    res = process_map(
        process_frame,
        zod_frames,
        repeat(anonymization),
        repeat(output_dir),
        repeat(train_frames),
        repeat(val_frames),
        desc="Processing frames",
        chunksize=100,
    )
    # extract the train/val paths
    train_paths = [r[0] for r in res if r[0] is not None]
    val_paths = [r[1] for r in res if r[1] is not None]

    # write the train/val lists
    train_output_file = os.path.join(
        output_dir,
        FILENAME.format(
            version=version, anonymization=anonymization, train_or_val=TRAIN
        ),
    )
    val_output_file = os.path.join(
        output_dir,
        FILENAME.format(version=version, anonymization=anonymization, train_or_val=VAL),
    )
    with open(train_output_file, "w") as f:
        f.write("\n".join(train_paths))
    with open(val_output_file, "w") as f:
        f.write("\n".join(val_paths))

    # print into
    print(f"Train list written to {train_output_file} with {len(train_paths)} frames")
    print(f"Val list written to {val_output_file} with {len(val_paths)} frames")


if __name__ == "__main__":
    typer.run(main)
