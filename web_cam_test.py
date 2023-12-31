import cv2
import torch
import numpy as np
from ultralytics import YOLO
from condition_check import burpees, cross_lunge, barbell_squat, side_lateral_raise, push_up, pull_up

pose_palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102], [230, 230, 0], [255, 153, 255],
                         [153, 204, 255], [255, 102, 255], [255, 51, 255], [102, 178, 255], [51, 153, 255],
                         [255, 153, 153], [255, 102, 102], [255, 51, 51], [153, 255, 153], [102, 255, 102],
                         [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0], [255, 255, 255]], dtype=np.uint8)
skeleton = [[22, 15], [15, 13], [13, 11], [23, 16], [16, 14], [14, 12],
            [21, 11], [21, 12], [21, 20], [20, 17],
            [17, 5], [17, 6], [5, 7], [6, 8], [7, 9], [8, 10], [9, 18], [10, 19],
            [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [0, 17]]  # , [3, 5], [4, 6]
limb_color = pose_palette[[9, 9, 9, 9, 9, 9, 7, 7, 7, 7,
                           0, 0, 0, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16]]  # , 16, 16
kpt_color = pose_palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0,
                          9, 9, 9, 9, 9, 9, 7, 0, 0, 7, 7, 9, 9]]

# Load YOLO model
# model = YOLO("yolov8m-pose.pt")
# model = YOLO("./runs/pose/train/weights/best1_10.pt")
# model = YOLO("./runs/pose/train/weights/best(2)(1).pt")
# model = YOLO("./runs/pose/train/weights/best2_4.pt")
# model = YOLO("./runs/pose/train/weights/best12.pt")
# model = YOLO("./runs/pose/train/weights/best13.pt")
model = YOLO("./runs/pose/train/weights/best14.pt")
# model = YOLO("./runs/pose/train/weights/best15.pt")


# Open the webcam (you can specify the camera index, 0 is usually the built-in webcam)
cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture("./asdf.mp4")
# cap = cv2.VideoCapture("./One-Minute Fitness Challenge Push-Ups.mp4")
# cap = cv2.VideoCapture("./pull_half.mp4")
# cap = cv2.VideoCapture("./pullupu.mp4")
# cap = cv2.VideoCapture("./pushuup.mp4")
# cap = cv2.VideoCapture("./pushup.mp4")


keyframe_interval = 1
frame_count = 0

while True:
    # Capture a frame from the webcam
    ret, frame = cap.read()

    if ret and frame_count % keyframe_interval == 0:
        h, w, c = frame.shape
        scale_factor_x = 640 / w
        scale_factor_y = 640 / h

        # Perform object detection using YOLO
        with torch.no_grad():
            results = model.predict(frame, save=False, imgsz=640, conf=0.5, device='cuda')[0]

        # print(results.keypoints)

        # Extract bounding boxes and keypoints from YOLO results
        boxes = results.boxes.xyxy
        cls = results.boxes.cls
        conf = results.boxes.conf
        cls_dict = results.names
        kps = results.keypoints.xy[0].cpu().numpy()

        # Resize the frame for visualization
        frame = cv2.resize(frame, (640, 640))

        # Draw bounding boxes and keypoints on the frame
        for box, cls_number, conf in zip(boxes, cls, conf):
            # conf_number = float(conf)
            cls_number_int = int(cls_number)
            cls_name = cls_dict[cls_number_int]
            x1, y1, x2, y2 = box
            x1_int = int(x1.item())
            y1_int = int(y1.item())
            x2_int = int(x2.item())
            y2_int = int(y2.item())

            x1_scale = int(x1_int * scale_factor_x)
            y1_scale = int(y1_int * scale_factor_y)
            x2_scale = int(x2_int * scale_factor_x)
            y2_scale = int(y2_int * scale_factor_y)

            frame = cv2.rectangle(frame, (x1_scale, y1_scale), (x2_scale, y2_scale), (0, 255, 0), 2)
            cv2.putText(frame, cls_name, (x1_scale, y1_scale), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # if results.keypoints.conf is not None:
        #     k_conf = results.keypoints.conf[0].cpu().numpy()
        #     for keypoint, conf in zip(kps, k_conf):
        #         x, y = int(keypoint[0] * scale_factor_x), int(keypoint[1] * scale_factor_y)
        #         cv2.circle(frame, (x, y), 2, (0, 255, 255), 2)
        #         cv2.putText(frame, str(conf)[:4], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # else:
        for i, keypoint in enumerate(kps):
            x, y = int(keypoint[0] * scale_factor_x), int(keypoint[1] * scale_factor_y)
            cv2.circle(frame, (x, y), 2, kpt_color[i].tolist(), 2)

        for j, (s, t) in enumerate(skeleton):
            try:
                x1, y1 = int(kps[s][0] * scale_factor_x), int(kps[s][1] * scale_factor_y)
                x2, y2 = int(kps[t][0] * scale_factor_x), int(kps[t][1] * scale_factor_y)
                cv2.line(frame, (x1, y1), (x2, y2), limb_color[j].tolist(), 2)
            except IndexError as e:
                continue

        # Display the frame with bounding boxes and keypoints
        # frame = cv2.resize(frame, (768, 768))
        cv2.imshow("Webcam YOLO Keypoint Detection", frame)

    frame_count += 1

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
