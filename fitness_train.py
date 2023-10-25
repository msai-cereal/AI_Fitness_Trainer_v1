from ultralytics import YOLO


if __name__ == "__main__":
    model = YOLO('yolov8m-pose.pt').to('cuda')
    model.train(data="./fitness.yaml", epochs=10, batch=32, lrf=0.001)

    # # resume
    # model = YOLO("./runs/detect/train3/weights/last.pt")
    # model.train(resume=True)