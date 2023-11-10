import cv2
import numpy as np
from numpy import degrees, arccos, dot
from numpy.linalg import norm
import time
from ultralytics import YOLO
# from PIL import ImageFont, ImageDraw, Image
# from gtts import gTTS
import os
import torch
import torch.nn as nn

pose_palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102], [230, 230, 0], [255, 153, 255],
                         [153, 204, 255], [255, 102, 255], [255, 51, 255], [102, 178, 255], [51, 153, 255],
                         [255, 153, 153], [255, 102, 102], [255, 51, 51], [153, 255, 153], [102, 255, 102],
                         [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0], [255, 255, 255]], dtype=np.uint8)
skeleton = [[22, 15], [15, 13], [13, 11], [23, 16], [16, 14], [14, 12],
            [21, 11], [21, 12], [21, 20], [20, 17],
            [17, 5], [17, 6], [5, 7], [6, 8], [7, 9], [8, 10], [9, 18], [10, 19],
            [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [0, 17], [3, 5], [4, 6]]
limb_color = pose_palette[[9, 9, 9, 9, 9, 9, 7, 7, 7, 7,
                           0, 0, 0, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16, 16]]
kpt_color = pose_palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0,
                          9, 9, 9, 9, 9, 9, 7, 0, 0, 7, 7, 9, 9]]

# Load YOLO model
model = YOLO("./runs/pose/train/weights/best14.pt")
# font_path = 'NotoSansKR-VariableFont_wght.ttf'  # 한글 폰트 파일 경로
# Open the webcam (you can specify the camera index, 0 is usually the built-in webcam)
# video_path = 'test_srl.mp4'
# 'test_bt.mp4'
# 'test_bs.mp4'
# 'test_cl.mp4'
# 'test_srl.mp4'
# 'test_psu.mp4'
# 'test_plu.mp4'
cap = cv2.VideoCapture(0)

keyframe_interval = 1
frame_count = 0

# 클래스 정보
keypoint_classes = {
    0: 'burpees',
    1: 'cross_lunge',
    2: 'barbell_squat',
    3: 'side_lateral_raise',
    4: 'push_up',
    5: 'pull_up',
    6: 'person',
    7: 'Ready'

}
cls_name = 'person'
start_bt_nose = 0
start_bt_Shoulder_R = 0
start_bt_Shoulder_L = 0
start_cl_nose = 0
start_cl_ankle_R = 0
start_cl_ankle_L = 0
start_cl_Knee_R = 0
start_cl_Knee_L = 0
start_cl_elbow_R = 0
start_cl_elbow_L = 0
start_bs_nose = 0
start_bs_waist = 0
start_bs_ankle = 0
start_bs_foot_R = 0
start_bs_foot_L = 0
start_srl_palm = 0
start_srl_Waist = 0
start_srl_Elbow_R = 0
start_srl_Elbow_L = 0
start_psu_nose = 0
start_psu_waist_R = 0
start_psu_waist_L = 0
start_psu_foot_R = 0
start_psu_foot_L = 0
start_plu_waist = 0

# 카운터 초기화
count_bt = 0
count_cl = 0
count_bs = 0
count_slr = 0
count_psu = 0
count_plu = 0

# 시작 조건 초기화
bt_condition_satisfied = False
cl_condition_satisfied = False
slr_condition_satisfied = False
bs_condition_satisfied = False
plu_condition_satisfied = False
psu_condition_satisfied = False

# 에러메세지 초기화
error_message = ""


def cal_angle(A, B, C):  # 각도
    angle = degrees(arccos(dot(A - B, C - B) / (norm(B - A) * norm(C - B))))
    return angle


def cal_distance(A, B):  # 거리
    distance = ((A[0] - B[0]) ** 2 + (A[1] - B[1]) ** 2) ** 0.5
    return distance


def cal_parallel(A, B, C, D):  # 평행

    slope1 = (B[1] - A[1]) / (B[0] - A[0])

    slope2 = (D[1] - C[1]) / (D[0] - C[0])

    return abs(slope1 - slope2) <= 5


def cal_x_equal(A, B, threshold=14):  # 같은 선상에 x좌표 값인지
    return abs(A[0] - B[0]) <= threshold


def cal_y_equal(A, B, threshold=5):  # 같은 선상에 y좌표 값인지
    return abs(A[1] - B[1]) <= threshold


# cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    if frame_count % keyframe_interval == 0:
        h, w, c = frame.shape
        scale_factor_x = 640 / w
        scale_factor_y = 640 / h

        results = model.predict(frame, save=False, imgsz=640, conf=0.5, device='cuda', verbose=False)[0]

        boxes = results.boxes.xyxy
        cls = results.boxes.cls
        conf = results.boxes.conf
        cls_dict = results.names
        kps = results.keypoints.xy[0].cpu().numpy()
        kps_xy = results.keypoints.xy[0][:, :2].cpu().numpy()
        for key_xy in kps:
            Nose = kps_xy[0]
            Left_Eye = kps_xy[1]
            Right_Eye = kps_xy[2]
            Left_Ear = kps_xy[3]
            Right_Ear = kps_xy[4]
            Left_Shoulder = kps_xy[5]
            Right_Shoulder = kps_xy[6]
            Left_Elbow = kps_xy[7]
            Right_Elbow = kps_xy[8]
            Left_Wrist = kps_xy[9]
            Right_Wrist = kps_xy[10]
            Left_Hip = kps_xy[11]
            Right_Hip = kps_xy[12]
            Left_Knee = kps_xy[13]
            Right_Knee = kps_xy[14]
            Left_Ankle = kps_xy[15]
            Right_Ankle = kps_xy[16]
            Neck = kps_xy[17]
            Left_Palm = kps_xy[18]
            Right_Palm = kps_xy[19]
            Back = kps_xy[20]
            Waist = kps_xy[21]
            Left_Foot = kps_xy[22]
            Right_Foot = kps_xy[23]

        frame = cv2.resize(frame, (640, 640))

        for box, _, conf in zip(boxes, keypoint_classes, conf):
            x1, y1, x2, y2 = box
            x1_scale = int(x1.item() * scale_factor_x)
            y1_scale = int(y1.item() * scale_factor_y)
            x2_scale = int(x2.item() * scale_factor_x)
            y2_scale = int(y2.item() * scale_factor_y)

            frame = cv2.rectangle(frame, (x1_scale, y1_scale), (x2_scale, y2_scale), (0, 255, 0), 2)

            # text = error_message
            # font = ImageFont.truetype(font_path, 40)
            # pil_image = Image.fromarray(frame)
            # draw = ImageDraw.Draw(pil_image)
            # draw.text((x1_scale, y1_scale-50), text, (0, 0, 255), font=font)
            # frame = np.array(pil_image)

            # 시작 규칙
            start_end_burpee_test = cal_parallel(Right_Shoulder, Left_Shoulder, Right_Elbow, Left_Elbow) and (
                        abs(Right_Shoulder[0] - Left_Shoulder[0]) - abs(Right_Ankle[0] - Left_Ankle[0])) >= 0 and (
                                                Nose[0] <= Left_Eye[0] <= Right_Ear[0] or Nose[0] >= Right_Eye[0] >=
                                                Left_Ear[0])
            start_end_cross_lunge = cal_parallel(Right_Shoulder, Left_Shoulder, Right_Elbow, Left_Elbow) and (
                        abs(Right_Ankle[0] - Left_Ankle[0]) - abs(Right_Ear[0] - Left_Ear[0])) >= 0 and cal_x_equal(
                Nose, Waist) and cal_x_equal(Waist, Back)
            start_end_barbell_squat = cal_parallel(Right_Shoulder, Left_Shoulder, Right_Elbow, Left_Elbow) and (
                        abs(Right_Shoulder[0] - Left_Shoulder[0]) - abs(Right_Ankle[0] - Left_Ankle[0])) >= 0 and (
                                                  Neck[1] - 5 <= min(Right_Palm[1], Left_Palm[1]) <= min(Right_Elbow[1],
                                                                                                         Left_Elbow[1]))
            start_end_side_lateral_raise = cal_parallel(Right_Shoulder, Left_Shoulder, Right_Elbow, Left_Elbow) and (
                        abs(Right_Ankle[0] - Left_Ankle[0]) - abs(Right_Ear[0] - Left_Ear[0])) >= 0 and (
                                                       cal_x_equal(Nose, Waist) and cal_x_equal(Waist, Back)) and max(
                Right_Palm[1], Left_Palm[1]) >= Waist[1]
            start_end_push_up = (Neck[1] <= max(Right_Foot[1], Left_Foot[1])) and min(Right_Ankle[1],
                                                                                      Left_Ankle[1]) <= max(
                Right_Palm[1], Left_Palm[1]) and min(Right_Ear[1], Left_Ear[1]) <= max(Right_Foot[1],
                                                                                       Left_Foot[1]) and abs(
                Right_Palm[0] - Left_Palm[0]) <= 3
            start_end_pull_up = cal_parallel(Right_Shoulder, Left_Shoulder, Right_Elbow, Left_Elbow) and min(
                Right_Palm[1], Left_Palm[1]) <= min(Right_Elbow[1], Left_Elbow[1]) <= Nose[1] <= min(Right_Shoulder[1],
                                                                                                     Left_Shoulder[
                                                                                                         1]) and (
                                            Nose[1] <= Neck[1] <= min(Right_Shoulder[1], Left_Shoulder[1]))

            # 버피 디버깅
            # cond1 = cal_parallel(Right_Shoulder,Left_Shoulder,Right_Elbow, Left_Elbow)
            # cond2 = (abs(Right_Shoulder[0]-Left_Shoulder[0])-abs(Right_Ankle[0]-Left_Ankle[0]))>=0
            # cond3 = (Nose[0]<=Left_Eye[0]<=Right_Ear[0] or Nose[0]>=Right_Eye[0]<=Left_Ear[0])

            # print(f"cond1: {cond1}, cond2: {cond2}, cond3: {cond3}")

            # start_end_bt = cond1 and cond2 and cond3
            # print(f"start_end_burpee_test: {start_end_burpee_test}")

            # 사레레 디버깅
            # cond1 = cal_parallel(Right_Shoulder, Left_Shoulder, Right_Elbow, Left_Elbow)
            # cond2 = (abs(Right_Ankle[0] - Left_Ankle[0]) - abs(Right_Ear[0] - Left_Ear[0])) >= 0
            # cond3 = cal_x_equal(Nose, Waist)

            # # 중간조건
            # cond4 = start_slr_palm <= max(Right_Palm[1],Left_Palm[1])
            # cond5 = (cal_x_equal(Nose,Waist) and cal_x_equal(Waist,Back))
            # cond6 = max(Left_Wrist[1], Right_Wrist[1]) >= Nose[1]
            # cond7 = (abs(Right_Palm[0]-Left_Palm[0])-abs(Right_Shoulder[0]-Left_Shoulder[0])>=2)
            # cond8 =  abs(Right_Elbow[1]-Left_Elbow[1])<=9

            # print(f"cond1: {cond1}, cond2: {cond2}, cond3: {cond3}, cond4: {cond4}, cond5 : {cond5}, cond6 : {cond6}, cond7:{cond7}, cond8: {cond8}")

            # start_end_side_lateral_raise = cond1 and cond2 and cond3 and cond4 and cond5 and cond6 and cond7 and cond8
            # print(f"start_end_side_lateral_raise: {start_end_side_lateral_raise}")

            # 크로스런지 디버깅
            # cond1 = cal_parallel(Right_Shoulder, Left_Shoulder, Right_Elbow, Left_Elbow)
            # cond2 = (abs(Right_Ankle[0]- Left_Ankle[0]) - abs(Right_Ear[0]- Left_Ear[0])) >=0
            # cond3 = cal_x_equal(Nose,Waist) and cal_x_equal(Waist,Back)

            # # 중간조건
            # cond4 =  (start_cl_ankle_R >= Right_Ankle[1]+1)
            # cond5 = (start_cl_ankle_L>= Left_Ankle[1]+1)
            # cond6 = start_cl_Knee_L>Left_Knee
            # cond7 = start_cl_Knee_R>Right_Knee
            # cond8 =  (start_cl_elbow_R <= Right_Elbow[1] or start_cl_elbow_L <= Left_Elbow[1])

            # print(f"cond1: {cond1}, cond2: {cond2}, cond3: {cond3}, cond4: {cond4}, cond5 : {cond5},cond4_5: {cond6}, cond5_5 : {cond7} cond6 : {cond8}")

            # start_end_side_lateral_raise = cond1 and cond2 and cond3 and cond4 and cond5 and cond6 and cond7 and cond8
            # print(f"start_end_side_lateral_raise: {start_end_side_lateral_raise}")

            # 푸시업 디버깅
            # cond1 = (Neck[1]<=min(Right_Foot[1],Left_Foot[1]))
            # cond2 = max(Right_Ankle[1],Left_Ankle[1])<=min(Right_Palm[1],Left_Palm[1])
            # cond3 =  max(Right_Ear[1],Left_Ear[1])<=min(Right_Foot[1],Left_Foot[1])
            # cond4 = abs(Right_Palm[0]-Left_Palm[0])<=3

            # print(f"cond1: {cond1}, cond2: {cond2}, cond3: {cond3}, cond4: {cond4}")

            # start_end_push_up = cond1 and cond2 and cond3 and cond4
            # print(f"start_end_push_up: {start_end_push_up}")

            # 바벨 스쿼트 디버깅
            # 시작 동작
            # cond1 = cal_parallel(Right_Shoulder,Left_Shoulder,Right_Elbow, Left_Elbow)
            # cond2 = (abs(Right_Shoulder[0]-Left_Shoulder[0])-abs(Right_Ankle[0]-Left_Ankle[0])) >=0
            # cond3 =  (Neck[1] -10 <= min(Right_Palm[1],Left_Palm[1]) <= min(Right_Elbow[1],Left_Elbow[1]))
            # # 중간동작
            # cond4 = (Nose[0]+ 10 >= Right_Ankle[0] >= Left_Ankle[0]) or (Nose[0]+10 <= Right_Ankle[0] <= Left_Ankle[0])
            # # cond5 =

            # print(f"cond1: {cond1}, cond2: {cond2}, cond3: {cond3}, cond4 : {cond4}" )

            # start_end_bs = cond1 and cond2 and cond3 and cond4
            # print(f"start_end_bs: {start_end_bs}")
            # 풀업 디버깅

            # cond1 = cal_parallel(Right_Shoulder,Left_Shoulder,Right_Elbow, Left_Elbow)
            # cond2 = min(Right_Palm[1],Left_Palm[1]) <= min(Right_Elbow[1],Left_Elbow[1]) <= Nose[1] <= min(Right_Shoulder[1],Left_Shoulder[1])
            # cond3 =  (Nose[1]<=Neck[1]<=min(Right_Shoulder[1],Left_Shoulder[1]))
            # # 중간동작
            # cond4 = min(Right_Palm[1],Left_Palm[1])<=min(Right_Shoulder[1],Left_Palm[1])
            # cond5 = (start_plu_waist >= Waist[1])

            # print(f"cond1: {cond1}, cond2: {cond2}, cond3: {cond3}, cond4 : {cond4}, cond5 : {cond5}" )

            # start_end_plu = cond1 and cond2 and cond3 and cond4 and cond5
            # print(f"start_end_plu: {start_end_plu}")

            if start_end_burpee_test:
                bt_condition_satisfied = True
                if start_bt_nose == 0:
                    start_bs_nose = Nose[1]
                if start_bt_Shoulder_R == 0:
                    start_bs_Shoulder_R = Right_Shoulder[1]
                if start_bt_Shoulder_L == 0:
                    start_bs_Shoulder_L = Left_Shoulder[1]

            if bt_condition_satisfied and start_bt_nose <= Nose[1] and start_bs_Shoulder_R <= Right_Shoulder[
                1] and start_bs_Shoulder_L <= Left_Shoulder[1] and max(Right_Shoulder[1], Left_Shoulder[1]) <= max(
                    Right_Palm[1], Left_Palm[1]):
                cls_name = 'burpee_test'

            if start_end_cross_lunge:
                cl_condition_satisfied = True
                if start_cl_nose == 0:
                    start_cl_nose = Nose[1]
                if start_cl_ankle_R == 0:
                    start_cl_ankle_R = Right_Ankle[1]
                if start_cl_ankle_L == 0:
                    start_cl_ankle_L = Left_Ankle[1]
                if start_cl_Knee_R == 0:
                    start_cl_Knee_R = Right_Knee[0]
                if start_cl_Knee_L == 0:
                    start_cl_Knee_L = Left_Knee[0]
                if start_cl_elbow_R == 0:
                    start_cl_elbow_R = Right_Elbow[1]
                if start_cl_elbow_L == 0:
                    start_cl_elbow_L = Left_Elbow[1]

            if cl_condition_satisfied and start_cl_nose <= Nose[1] and abs(Nose[1] - start_cl_nose) <= 10 and (
                    (start_cl_ankle_R >= Right_Ankle[1] + 1) or (start_cl_ankle_L >= Left_Ankle[1] + 1)) and (
                    start_cl_elbow_R <= Right_Elbow[1] or start_cl_elbow_L <= Left_Elbow[1]) and Back[1] <= max(
                    Right_Palm[1], Left_Palm[1]) and (
                    start_cl_Knee_R < Right_Knee[0] or start_cl_Knee_L > Left_Knee[0]):
                cls_name = 'cross_lunge'

            if start_end_barbell_squat:
                bs_condition_satisfied = True
                if start_bs_nose == 0:
                    start_bs_nose = Nose[1]
                if start_bs_waist == 0:
                    start_bs_waist = Waist[1]
                if start_bs_ankle == 0:
                    start_bs_ankle = max(Right_Ankle[1], Left_Ankle[1])
                if start_bs_foot_R == 0:
                    start_bs_foot_R = Right_Foot[1]
                if start_bs_foot_L == 0:
                    start_bs_foot_L = Left_Foot[1]
            if bs_condition_satisfied and start_bs_nose < Nose[1] and abs(
                    start_bs_ankle - max(Right_Ankle[1], Left_Ankle[1])) <= 2 and (
                    (Nose[0] + 10 >= Right_Ankle[0] >= Left_Ankle[0]) or (
                    Nose[0] + 10 <= Right_Ankle[0] <= Left_Ankle[0])) and ((start_bs_foot_R - Right_Foot[1]) <= 2 or (
                    start_bs_foot_L - Left_Foot[1]) <= 2) and start_bs_waist >= Neck[1] and (
                    Neck[1] - 5 <= min(Right_Palm[1], Left_Palm[1]) <= min(Right_Elbow[1], Left_Elbow[1])):
                cls_name = 'barbell_squat'

            if start_end_side_lateral_raise:
                slr_condition_satisfied = True
                if start_srl_palm == 0:
                    start_srl_palm = max(Right_Palm[1], Left_Palm[1])
                if start_srl_Waist == 0:
                    start_srl_Waist = Waist[1]
                if start_srl_Elbow_R == 0:
                    start_srl_Elbow_R = Right_Elbow[0]
                if start_srl_Elbow_L == 0:
                    start_srl_Elbow_L = Left_Elbow[0]

            # and abs(Nose[1]-start_slr_nose)<=2
            # and start_slr_palm <= max(Right_Palm[1],Left_Palm[1])
            if slr_condition_satisfied and max(Left_Wrist[1], Right_Wrist[1]) >= Nose[1] and (
                    abs(Right_Palm[0] - Left_Palm[0]) - abs(Right_Shoulder[0] - Left_Shoulder[0]) >= 2) and abs(
                    Waist[1] - start_srl_Waist) <= 6 and (
                    (abs(Left_Elbow[0] - Right_Elbow[0]) - abs(start_srl_Elbow_L - start_srl_Elbow_R)) >= 0) and (
                    (abs(Left_Elbow[0] - Right_Elbow[0]) - abs(Left_Palm[0] - Right_Palm[0])) <= 0):
                cls_name = "side_lateral_raise"

            if start_end_push_up:
                psu_condition_satisfied = True
                if start_psu_nose == 0:
                    start_psu_nose = Nose[1]
                if start_psu_waist_R == 0:
                    start_psu_waist_R = Right_Wrist[0]
                if start_psu_waist_L == 0:
                    start_psu_waist_L = Left_Wrist[0]
                if start_psu_foot_R == 0:
                    start_psu_foot_R = Left_Wrist[0]
                if start_psu_foot_L == 0:
                    start_psu_foot_L = Left_Wrist[0]
            if psu_condition_satisfied and start_psu_nose <= Nose[1] and (
                    (start_psu_waist_R - Right_Wrist[0]) <= 2 or (start_psu_waist_L - Left_Wrist[0]) <= 2) and (
                    (start_psu_foot_R - Right_Palm[0]) <= 2 or (start_psu_foot_L - Left_Palm[0]) <= 2):
                cls_name = 'push_up'

            if start_end_pull_up:
                plu_condition_satisfied = True
                if start_plu_waist == 0:
                    start_plu_waist = Waist[1]
            if start_end_pull_up and min(Right_Palm[1], Left_Palm[1]) <= Back[1] and start_plu_waist >= Waist[1]:
                cls_name = 'pull_up'

            else:
                error_message = "Ready"
                # error_message += "시작자세가 바르지 않습니다. 정자세로 준비해주세요."

            cv2.putText(frame, cls_name, (x1_scale, y1_scale - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, error_message, (150, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # print(start_end_point_bt)
            # print(start_end_point_cl)
            # print(start_end_point_bs)
            # print(start_end_point_slr)
            # print(start_end_point_psu)
            # print(start_end_point_plu)

            # tts = gTTS(error_message, lang='ko')  # 'ko'는 한국어를 나타냅니다. 필요한 언어 코드로 변경하세요.

            # # 음성 파일 저장
            # tts.save("error_message.mp3")

            # # 음성 재생 (외부 음악 플레이어로 재생됩니다)
            # os.system("vlc error_message.mp3")  # window 사용자의 경우

            # # 음성 파일 삭제
            # os.remove("error_message.mp3")

            # if cls_name == 'burpee_test':
            #     if Waist[1] <= Nose[1]:
            #         error_message += ""
            #         if cal_angle(Right_Hip,Right_Knee,Right_Ankle) <= 90:
            #             error_message += ""
            #             if max(Right_Ankle[1],Left_Ankle[1]) <= max(Right_Hip[1],Left_Hip[1]) <= max(Right_Ear[1],Left_Ear[1]) <= Nose[1] and (cal_angle(Right_Wrist,Right_Elbow,Right_Shoulder) <= 60 or cal_angle(Left_Wrist,Left_Elbow,Left_Shoulder) <=60) and (max(Right_Shoulder[0],Left_Shoulder[0])>= max(Right_Wrist[0],Left_Wrist[0])>=max(Right_Elbow[0],Left_Elbow[0]) or max(Right_Shoulder[0],Left_Shoulder[0])<=max(Right_Wrist[0],Left_Wrist[0])<=max(Right_Elbow[0],Left_Elbow[0])):
            #                 error_message += ""
            #                 # if start_end_point_bt == start_end_point_bt:
            #                 #     count_bt +=1
            #                 #     error_message += ""
            #                 # else:
            #                 #     error_message += "처음 시작자세와 동일하게 한동작을 마무리하세요. 횟수가 카운터 되지 않습니다"
            #             else :
            #                 error_message += "자세를 더 낮추세요"
            #         else:
            #             error_message += "다리를 더 오므리세요."
            #     else:
            #         error_message += "허리를 굽히세요."

            # elif cls_name == 'cross_lunge':
            #     if cal_x_equal(Nose,Right_Ankle) and cal_x_equal(Nose,Left_Ankle):
            #         error_message += ""
            #         if abs(Right_Ankle[0]-Left_Ankle[0]) - abs(Right_Ear[0]-Left_Ear[0])>=0:
            #             error_message += ""
            #             if cal_parallel(Right_Shoulder,Left_Shoulder,Right_Elbow, Left_Elbow):
            #                 error_message += ""
            #                 # if start_end_point_cl == start_end_point_cl:
            #                 #     count_cl +=1
            #                 #     error_message += ""
            #             else:
            #                 error_message += "몸이 휘지 않도록 해주세요"
            #         else:
            #             error_message += "발이 양쪽으로 더 벌어지도록 해주세요"
            #     else:
            #         error_message += "무릎의 위치가 일치시켜주세요"
            # elif cls_name == 'barbell_squat':
            #     if cal_parallel(Right_Knee,Left_Knee,Right_Ankle,Left_Ankle):
            #         error_message += ""
            #         if max(Right_Knee[1],Left_Knee[1]) <= max(Right_Hip[1],Left_Hip[1]):
            #             error_message += ""
            #             if (Nose[0] >= max(Right_Ankle[0],Left_Ankle[0])) or (Nose[0] <= min(Right_Ankle[0],Left_Ankle[0])):
            #                 error_message += ""
            #                 # if start_end_point_bs == start_end_point_bs:
            #                 #     count_bs +=1
            #                 #     error_message += ""
            #             else:
            #                 error_message += "허리를 더 숙이고 정면을 바라봐주세요"
            #         else:
            #             error_message += "엉덩이를 더 내려주세요"
            #     else:
            #         error_message += "무릎과 발목의 방향을 일치시켜주세요"
            # elif cls_name == 'side_lateral_raise':
            #     if cal_angle(Waist,Neck,Right_Elbow)==90:
            #         if cal_y_equal(Right_Elbow,Left_Elbow):
            #             if Nose[1]<=max(Right_Palm[1],Left_Palm[1]):
            #                 error_message += "팔이 너무 올라갔습니다"
            #                 # if start_end_point_slr == start_end_point_slr:
            #                 #     count_slr +=1
            #                 #     error_message += ""
            #         else:
            #             error_message += "양팔의 높이를 같게 해주세요"
            #     else:
            #         error_message += "팔을 더 올려주세요"

            # elif cls_name == 'push_up':
            #     if (cal_angle(Right_Shoulder,Right_Elbow,Right_Wrist) <= 105 or cal_angle(Left_Shoulder,Left_Elbow,Left_Wrist) <= 105):
            #         error_message += ""
            #         if cal_y_equal(Right_Ear,Right_Ankle) or cal_y_equal(Left_Ear,Left_Ankle):
            #             error_message += ""
            #             # if start_end_point_psu == start_end_point_psu:
            #             #         count_psu +=1
            #             #         error_message += ""
            #         else:
            #             error_message += "허리를 펴고, 엉덩이를 올리지말고, 고개를 숙이지 마세요 "
            #     else:
            #         error_message += "팔을 더 굽혀주세요"
            # elif cls_name == 'pull_up':
            #     if Neck[1] <= max(Right_Palm[1],Left_Palm[1]):
            #         error_message += ""
            #         if cal_parallel(Right_Knee,Left_Knee,Right_Ankle,Left_Ankle):
            #             error_message += ""
            #             # if start_end_point_plu == start_end_point_plu:
            #             #         count_plu +=1
            #             #         error_message += ""
            #         else:
            #             error_message += "팔이 한쪽으로 휘지 않도록 하세요"
            #     else:
            #         error_message += "더 올라가주세요"

        for keypoint in kps:
            x, y = int(keypoint[0] * scale_factor_x), int(keypoint[1] * scale_factor_y)
            cv2.circle(frame, (x, y), 2, (0, 255, 255), 2)

        for j, (s, t) in enumerate(skeleton):
            try:
                x1, y1 = int(kps[s][0] * scale_factor_x), int(kps[s][1] * scale_factor_y)
                x2, y2 = int(kps[t][0] * scale_factor_x), int(kps[t][1] * scale_factor_y)
                cv2.line(frame, (x1, y1), (x2, y2), limb_color[j].tolist(), 2)
            except IndexError as e:
                continue

    cv2.imshow('YOLOv8-pose 키포인트 검출', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()