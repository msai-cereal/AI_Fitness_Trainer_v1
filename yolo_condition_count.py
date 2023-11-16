import cv2
import torch
import numpy as np
from ultralytics import YOLO
from condition_check import burpees, pull_up, cross_lunge, side_lateral_raise, barbell_squat, push_up
from countings import count_burpees, count_pull_up, count_cross_lunge, count_side_lateral_raise, count_barbell_squat, \
    count_push_up

check_fns = {0: burpees, 1: pull_up, 2: cross_lunge, 3: side_lateral_raise, 4: barbell_squat, 5: push_up}
count_fns = {0: count_burpees, 1: count_pull_up, 2: count_cross_lunge, 3: count_side_lateral_raise,
             4: count_barbell_squat, 5: count_push_up}
exercise = {0: 'burpees', 1: 'pull_up', 2: 'cross_lunge', 3: 'side_lateral_raise', 4: 'barbell_squat', 5: 'push_up'}

pose_palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102], [230, 230, 0], [255, 153, 255],
                         [153, 204, 255], [255, 102, 255], [255, 51, 255], [102, 178, 255], [51, 153, 255],
                         [255, 153, 153], [255, 102, 102], [255, 51, 51], [153, 255, 153], [102, 255, 102],
                         [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0], [255, 255, 255]], dtype=np.uint8)
skeleton = [[22, 15], [15, 13], [13, 11], [23, 16], [16, 14], [14, 12],
            [21, 11], [21, 12], [21, 20], [20, 17],
            [17, 5], [17, 6], [5, 7], [6, 8], [7, 9], [8, 10], [9, 18], [10, 19]]
# [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [0, 17], [3, 5], [4, 6]]      # 눈 코 귀 시각화 x
limb_color = pose_palette[[9, 9, 9, 9, 9, 9, 7, 7, 7, 7,
                           0, 0, 0, 0, 0, 0, 0, ]]  # , 16, 16, 16, 16, 16, 16, 16, 16]]
kpt_color = pose_palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0,
                          9, 9, 9, 9, 9, 9, 7, 0, 0, 7, 7, 9, 9]]

# model = YOLO("./runs/pose/train/weights/best14.pt")  # pt 경로
model = YOLO("./runs/pose/train/weights/best21.pt")  # pt 경로

cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture("비디오 경로")

keyframe_interval = 1
frame_count = 0
sequence = []
flip_sequence = []
data = []
frames_per_exercise = 16  # condition_check에 집어넣을 시퀀스 단위
keypoints_per_frame = 24

# exercise = {0: 'burpees', 1: 'pull_up', 2: 'cross_lunge', 3: 'side_lateral_raise', 4: 'barbell_squat', 5: 'push_up'}
num_class = 3  # 사용자 선택 0 ~ 5
ep = exercise[num_class]
active = False  # 운동하는 상태  ->  나중에 휴식이나 정지 기능 구현할 때 손 봐야합니다.

messages = {""}
counts = 0
flag = False
predicted_class = None

while True:
    # Capture a frame from the webcam
    ret, frame = cap.read()

    if not ret:
        break

    if frame_count % keyframe_interval == 0:  # keyframe_interval 장마다 실행
        h, w, c = frame.shape

        with torch.no_grad():
            results = model.predict(frame, save=False, imgsz=640, conf=0.5, device='cuda', verbose=False)[0]

        kpts = results.keypoints.xy[0].cpu().numpy()

        # keypoints 리스트가 비어있는지 확인
        if len(kpts) == 0:
            # print("No person detected in the frame.")
            frame_count += 1
            # Display the resulting frame
            cv2.putText(frame, ep + f" {counts}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.imshow('Frame', frame)
            continue

        for i, keypoint in enumerate(kpts):
            # 눈 코 귀 시각화 x
            if i < 5:
                continue

            x, y = int(keypoint[0]), int(keypoint[1])
            cv2.circle(frame, (x, y), 2, kpt_color[i].tolist(), 2)

        for j, (s, t) in enumerate(skeleton):
            try:
                x1, y1 = int(kpts[s][0]), int(kpts[s][1])
                x2, y2 = int(kpts[t][0]), int(kpts[t][1])
                cv2.line(frame, (x1, y1), (x2, y2), limb_color[j].tolist(), 2)
            except IndexError as e:
                continue

        if frame_count % 3 == 0:  # webcam>> 3, video >> 9
            # 각 키포인트를 정규화
            _kpts_normalized = [[x / w, y / h] for x, y in kpts]
            # 시퀀스에 키포인트 추가
            data.append(_kpts_normalized)

        # 시퀀스가 충분히 쌓였는지 확인
        if len(data) == frames_per_exercise:
            try:
                active = True
                # predicted_class = 2  # 사용자가 운동을 선택하면, 이런 식으로 특정 운동의 함수로만 계산 진행
                messages = check_fns[num_class]([data])[0]
                for i, m in enumerate(messages, 2):
                    cv2.putText(frame, m, (10, 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
            except Exception as e:
                print(f"예외: , {e}\n" * 10)

            # 시퀀스 초기화
            data = []

        elif messages:
            for i, m in enumerate(messages, 2):
                cv2.putText(frame, m, (10, 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)

        if active:
            count, flag = count_fns[num_class]([[x / w, y / h] for x, y in kpts], flag)
            counts += count

    cv2.putText(frame, ep + f" {counts}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    # Display the resulting frame
    cv2.imshow('Frame', frame)
    frame_count += 1

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print(frame_count)
cap.release()
cv2.destroyAllWindows()
