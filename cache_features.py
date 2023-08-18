from models.yolo_encoder import Encoder


def main():
    cfg = "cfg/yolo_encoder.yaml"

    enc = Encoder(cfg)

    # load the weights from checkpoint

    print("here")


if __name__ == "__main__":
    main()
