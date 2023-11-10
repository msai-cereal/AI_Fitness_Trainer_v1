from ultralytics import YOLO

if __name__ == "__main__":
    # model = YOLO('yolov8m-pose.pt')
    model = YOLO('./runs/pose/train18/weights/best.pt')
    # model.train(data="./fitness.yaml", epochs=20, batch=32, lrf=0.001)
    #
    model.train(data="./fitness.yaml", epochs=10, batch=32, optimizer='AdamW', lrf=0.001, lr0=0.001,
                degrees=3, perspective=0.0001)

    # resume
    # model = YOLO("./runs/pose/train11/weights/last.pt")  # SGD, perspective=0.001
    # model.train(resume=True)
