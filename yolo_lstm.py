import cv2
import torch
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model
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
# [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [0, 17], [3, 5], [4, 6]]
limb_color = pose_palette[[9, 9, 9, 9, 9, 9, 7, 7, 7, 7,
                           0, 0, 0, 0, 0, 0, 0, ]]  # , 16, 16, 16, 16, 16, 16, 16, 16]]
kpt_color = pose_palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0,
                          9, 9, 9, 9, 9, 9, 7, 0, 0, 7, 7, 9, 9]]

# 예시를 위한 클래스별 이모지 매핑
class_to_emoji = {
    0: "🏋️‍♂️" * 10,  # Emoji for burpee_test
    1: "🧗" * 10,  # Emoji for pull_up
    2: "🤸‍♀️" * 10,  # Emoji for cross_lunge
    3: "🏋️" * 10,  # Emoji for side_lateral_raise
    4: "🏋️‍♀️" * 10,  # Emoji for barbell_squat
    5: "pp" * 10
}

model = YOLO("./runs/pose/train/weights/best14.pt")  # pt 경로
# model = YOLO("./runs/pose/train/weights/best20.pt")  # pt 경로
# lstm_model = load_model('./lstm_model_v43.h5')  # LSTM 모델의 파일 경로
lstm_model = load_model('./lstm_model_v200.h5')  # LSTM 모델의 파일 경로

# https://colab.research.google.com/drive/1bSqaOr7FOpPU-h7Xq4BaLazn1EsPm0Fp#scrollTo=FUanR4jJkg9R


cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture("비디오 경로")
# cap = cv2.VideoCapture("./videos/바벨스쿼트.mp4")
# cap = cv2.VideoCapture("./videos/버피.mp4")
# cap = cv2.VideoCapture("./videos/사레레b.mp4")
# cap = cv2.VideoCapture("./videos/크로스런지.mp4")
# cap = cv2.VideoCapture("./videos/푸쉬업.mp4") # 20
# cap = cv2.VideoCapture("./videos/풀업.mp4")   # x

keyframe_interval = 1
frame_count = 0
# num_sequence = 16
sequence = []
flip_sequence = []
data = []
frames_per_exercise = 16  # lstm_model_v43.h5 >> 9 / lstm_model_v200.h5 >> 16
keypoints_per_frame = 24
ep = '?????'
messages = {""}
counts = 0
flag = False
predicted_class = None

while True:
    # Capture a frame from the webcam
    ret, frame = cap.read()
    # if frame_count < 3900:
    #     frame_count += 1
    #     continue

    if not ret:
        break

    if frame_count % keyframe_interval == 0:  # keyframe_interval 장마다 실행
        h, w, c = frame.shape

        with torch.no_grad():
            results = model.predict(frame, save=False, imgsz=640, conf=0.5, device='cuda', verbose=False)[0]

        kpts = results.keypoints.xy[0].cpu().numpy()
        # print(kpts)
        # keypoints 리스트가 비어있는지 확인
        if len(kpts) == 0:
            # print("No person detected in the frame.")
            frame_count += 1
            # Display the resulting frame
            cv2.putText(frame, ep + f" {counts}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.imshow('Frame', frame)
            continue

        for i, keypoint in enumerate(kpts):
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

        if frame_count % 9 == 0:  # lstm_model_v200.h5: webcam>> 3, video >> 5
            # 각 키포인트를 정규화
            kpts_normalized = np.array([[x / w * 2 - 1, y / h * 2 - 1] for x, y in kpts])
            flip_kpts_normalized = np.array([[-x / w * 2 - 1, y / h * 2 - 1] for x, y in kpts])
            _kpts_normalized = [[x / w, y / h] for x, y in kpts]

            # 시퀀스에 키포인트 추가
            sequence.append(kpts_normalized.flatten())
            flip_sequence.append(flip_kpts_normalized.flatten())
            data.append(_kpts_normalized)

        # 시퀀스가 충분히 쌓였는지 확인
        if len(sequence) == frames_per_exercise:
            # 시퀀스를 배열로 변환하고 모델에 맞는 형태로 재구성
            sequence_array = np.array(sequence).reshape(1, frames_per_exercise, keypoints_per_frame * 2)

            # 예측 수행
            predictions = lstm_model.predict(sequence_array)
            if np.argmax(predictions) != 5:
                flip_sequence_array = np.array(flip_sequence).reshape(1, frames_per_exercise, keypoints_per_frame * 2)
                flip_predictions = lstm_model.predict(flip_sequence_array)
                if np.argmax(flip_predictions) == 5:
                    predictions = flip_predictions
            # 여기에서 predictions를 사용하여 무엇인가를 할 수 있습니다.
            # 예를 들어, 예측된 클래스를 출력하거나 다음 작업을 위해 저장할 수 있습니다.
            print(predictions)
            predicted_class = np.argmax(predictions)
            emoji_to_display = class_to_emoji.get(predicted_class, "❓")  # 기본값으로 물음표 이모지 설정

            # 이모지와 함께 예측된 클래스 출력
            print(f"Predicted Class: {predicted_class} {emoji_to_display}")
            tmp = ep
            ep = exercise[predicted_class]
            # ep = 'cross_lunge'
            if ep != tmp:
                counts = 0
                flag = False

            if predicted_class in check_fns:
                try:
                    # predicted_class = 2  # 사용자가 운동을 선택하면, 이런 식으로 특정 운동의 함수로만 계산 진행
                    messages = check_fns[predicted_class]([data])[0]
                    for i, m in enumerate(messages, 2):
                        cv2.putText(frame, m, (10, 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
                except Exception as e:
                    print(f"예외: , {e}\n" * 10)

            # 시퀀스 초기화
            sequence = []
            flip_sequence = []
            data = []

        elif messages:
            for i, m in enumerate(messages, 2):
                cv2.putText(frame, m, (10, 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)

        if predicted_class in range(6):
            count, flag = count_fns[predicted_class]([[x / w, y / h] for x, y in kpts], flag)
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

# import cv2
# cap = cv2.VideoCapture(0)
# if cap.isOpened:
#     file_path = '../CV2/video/record.mp4'
#     fps = 25.40
#     fourcc = cv2.VideoWriter_fourcc(*'DIVX')            # 인코딩 포맷 문자
#     width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
#     height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
#     size = (int(width), int (height))                   # 프레임 크기

#     out = cv2.VideoWriter(file_path, fourcc, fps, size) # VideoWriter 객체 생성
#     while True:
#         ret, frame = cap.read()
#         if ret:
#             cv2.imshow('camera-recording', frame)
#             out.write(frame)                            # 파일 저장
#             if cv2.waitKey(int(1000/fps)) != -1:
#                 break
#         else:
#             print('no file!')
#             break
#     out.release()                                       # 파일 닫기
# else:
#     print("Can`t open camera!")
# cap.release()
# cv2.destroyAllWindows()