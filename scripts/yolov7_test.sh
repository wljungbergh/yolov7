#!/bin/bash
#
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres=gpu:1
#SBATCH --output /workspaces/%u/yolov7logs/slurm-%j-run.out
#SBATCH --partition zprod

singularity exec --bind /datasets:/datasets --bind /staging:/staging --bind /workspaces:/workspaces \
    --nv \
    --bind /workspaces/s0001387/yolov7/coco:/coco/ \
    --pwd /workspaces/s0001387/yolov7 \
    /staging/agp/geohe/containers/yolov7.sif \
    python -u test.py --data data/densepose.yaml --img 640 --batch 32 --conf 0.001 --iou 0.65 --device 0 --weights /workspaces/s0001387/yolov7/runs/train/yolov7-custom/weights/best.pt --name yolov7_640_val
