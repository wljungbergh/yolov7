# How to run face anonymization
## Detection / Inference
### Step one
Download the trained weights (attached in the submitted zip-file).

### Step two
Build the Dockerfile by
```
cd docker
docker build -t face_anonymization:latest .
```

### Step three
Run the face anonymization script

```
cd <project-root>
docker run --gpus all  face_anonymization:latest python3 detect.py --weights <path-to-downloaded-wieghts> --source <image or .mp4 file> --blur
```

Note that you can use your webcam by setting `--source 0`.
If you wish to only see bounding boxes, leave out the `--blur` flag

### Step four
Enjoy life as a person whose privacy has remained intact.

## Training
We created the `create_labels_and_datalists.py` to preprocess the annotations into the correct format required by the `yolov7` framework. Note that this have been designed to work with the dataset structure that we have on our compute platform. This might need some tweaking if used
