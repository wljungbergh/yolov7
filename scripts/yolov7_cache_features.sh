#!/bin/bash
#
#SBATCH --nodes 1
#SBATCH --gres=gpu:1
#SBATCH -c 32
#SBATCH --time 14-00:00:00
#SBATCH --output /workspaces/%u/yolov7logs/slurm-%j-run.out
#SBATCH --partition zprod

REPO_ROOT=/workspaces/s0001038/repos/yolov7

singularity exec --bind /datasets:/datasets --bind /staging:/staging --bind /workspaces:/workspaces \
    --nv \
    --pwd $REPO_ROOT \
    /staging/agp/geohe/containers/yolov7.sif \
    python -u cache_features.py --workers 8\
     --cfg $REPO_ROOT/cfg/yolo_encoder.yaml\
     --device 0\
     --batch-size 4\
     --data $REPO_ROOT/data/zod_original.yaml\
     --img 1920\
     --out-folder "/staging/agp/willju/zod_feature_cache"\
     --task test\