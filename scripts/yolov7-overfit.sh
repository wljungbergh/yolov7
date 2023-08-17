#!/bin/bash
#
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres=gpu:1
#SBATCH --time 3-00:00:00
#SBATCH --output /workspaces/%u/yolov7logs/slurm-%j-run.out
#SBATCH --partition ztestpreemp

singularity exec --bind /datasets:/datasets --bind /staging:/staging --bind /workspaces:/workspaces \
    --nv \
    --pwd /workspaces/s0001387/repos/yolov7 \
    /staging/agp/geohe/containers/yolov7.sif \
    python -u train_aux.py --workers 32 --weights "" --device 0 --adam --batch-size 16 --cache-images --data /workspaces/s0001387/repos/yolov7/data/zod_mini.yaml --nosave --img 1920 --noautoanchor --epochs 100000 --cfg cfg/training/yolov7_zod.yaml --name zod-overfit --hyp data/hyp.scratch.custom.yaml