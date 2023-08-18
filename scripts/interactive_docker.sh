REPO_ROOT="/yolov7"
DOCKER_NAME="yolov7:latest"
docker run -w $REPO_ROOT -it --gpus all --shm-size=8g -v `pwd`:$REPO_ROOT -v /staging:/staging -v /datasets:/datasets $DOCKER_NAME