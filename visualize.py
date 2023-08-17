import os
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

ZOD_CLASS_MAP = {
    "Vehicle": 0,
    "VulnerableVehicle": 1,
    "Pedestrian": 2,
}
class_id_to_name_mapping = {v: k for k, v in ZOD_CLASS_MAP.items()}


def plot_bounding_box(image, annotation_list):
    annotations = np.array(annotation_list)
    w, h = image.size

    plotted_image = ImageDraw.Draw(image)

    transformed_annotations = np.copy(annotations)
    transformed_annotations[:, [1, 3]] = annotations[:, [1, 3]] * w
    transformed_annotations[:, [2, 4]] = annotations[:, [2, 4]] * h

    transformed_annotations[:, 1] = transformed_annotations[:, 1] - (
        transformed_annotations[:, 3] / 2
    )
    transformed_annotations[:, 2] = transformed_annotations[:, 2] - (
        transformed_annotations[:, 4] / 2
    )
    transformed_annotations[:, 3] = (
        transformed_annotations[:, 1] + transformed_annotations[:, 3]
    )
    transformed_annotations[:, 4] = (
        transformed_annotations[:, 2] + transformed_annotations[:, 4]
    )

    for ann in transformed_annotations:
        obj_cls, x0, y0, x1, y1 = ann
        plotted_image.rectangle(((x0, y0), (x1, y1)))

        plotted_image.text((x0, y0 - 10), class_id_to_name_mapping[(int(obj_cls))])

    plt.imshow(np.array(image))
    plt.show()


annotations = [os.path.join("zod/labels", f) for f in os.listdir("zod/labels")]

# Get any random annotation file
annotation_file = random.choice(annotations)
with open(annotation_file, "r") as file:
    annotation_list = file.read().split("\n")[:-1]
    annotation_list = [x.split(" ") for x in annotation_list]
    annotation_list = [[float(y) for y in x] for x in annotation_list]

# Get the corresponding image file
# image_file = annotation_file.replace("annotations", "images").replace("txt", "png")
anno_basename = os.path.basename(annotation_file)
id_ = anno_basename.split("_")[0]
image_file = os.path.join(
    "/staging/dataset_donation/round_2/single_frames/",
    id_,
    "camera_front_original",
    anno_basename.replace("txt", "png"),
)
assert os.path.exists(image_file)

# Load the image
image = Image.open(image_file)

# Plot the Bounding Box
plot_bounding_box(image, annotation_list)
