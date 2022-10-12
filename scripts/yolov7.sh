#!/usr/bin/env bash
#SBATCH -t 0-16:00:00
#SBATCH -N 1

singularity exec --bind /datasets:/datasets --bind /staging:/staging --bind /workspaces:/workspaces \
    --nv \
    --bind /workspaces/s0001387/yolov7/coco:/coco/ \
    --pwd /workspaces/s0001387/yolov7 \
    /staging/agp/geohe/containers/yolov7.sif \
    python -u train.py --workers 8 --device 0 --batch-size 32 --data data/custom.yaml --img 640 640 --cfg cfg/training/yolov7-custom.yaml --weights 'yolov7_training.pt' --name yolov7-custom --hyp data/hyp.scratch.custom.yaml