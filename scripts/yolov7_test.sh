#!/bin/bash
#
#SBATCH --nodes 1
#SBATCH --gres=gpu:1
#SBATCH -c 32
#SBATCH --time 14-00:00:00
#SBATCH --output /workspaces/%u/yolov7logs/slurm-%j-run.out
#SBATCH --partition zprod

singularity exec --bind /datasets:/datasets --bind /staging:/staging --bind /workspaces:/workspaces \
    --nv \
    --pwd /workspaces/s0001387/repos/yolov7 \
    /staging/agp/geohe/containers/yolov7.sif \
     --weights runs/train/$1/weights/best.pt\
     --device 0\
     --batch-size 4\
     --data /workspaces/s0001387/repos/yolov7/data/zod_blur.yaml\
     --img 1920\
     --name $1\
     --task test\
     --verbose\
     --save-json\
     --exist-ok