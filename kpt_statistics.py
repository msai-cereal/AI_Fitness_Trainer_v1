from ultralytics import YOLO
import os
import glob
import json
from tqdm import tqdm
import numpy as np
from numpy import degrees, arccos, dot
from numpy.linalg import norm
import json
import pandas as pd

model = YOLO("./runs/pose/train15/weights/best.pt")

root_path = "C:/Users/labadmin/Downloads/fitness/train/labels"
jsons_list = glob.glob(os.path.join(root_path, '*', '*', '*.json'))

# six_exercise = {'049': 'burpees', '177': 'cross_lunge', '313': 'barbell_squat',
#                 '337': 'side_lateral_raise', '561': 'push_up', '713': 'pull_up', }
six_exercise = {'049': [], '177': [], '313': [], '337': [], '561': [], '713': [], }
part_list = ['Nose', 'Left Eye', 'Right Eye', 'Left Ear', 'Right Ear', 'Left Shoulder', 'Right Shoulder', 'Left Elbow',
             'Right Elbow', 'Left Wrist', 'Right Wrist', 'Left Hip', 'Right Hip', 'Left Knee', 'Right Knee',
             'Left Ankle', 'Right Ankle', 'Neck', 'Left Palm', 'Right Palm', 'Back', 'Waist', 'Left Foot', 'Right Foot']


cam = 'view3'
image_width = 1920
image_height = 1080
normalize_array = np.array([image_width, image_height])

for json_src in tqdm(jsons_list, leave=False):
    json_name = json_src.split("\\")[-1].split('.')[0]
    check = json_name.split('-')[-1]
    if check not in six_exercise:
        continue

    with open(json_src, 'r', encoding='utf-8') as f:
        annotations = json.load(f)

    data = []
    frames = annotations['frames']
    type_num = int(annotations['type'])
    for frame in frames:
        view = frame[cam]
        img_name = view['img_key'].split('/')[-1]
        pts = view['pts']
        kpt = [list(keypoint.values()) for keypoint in pts.values()]
        _part_list = list(pts.keys())
        if part_list != _part_list:
            print(json_src.split("\\")[-1], img_name)
            print(_part_list)
        data.append(kpt)
    data = np.array([data])
    # print(data.shape) # data.shape = (1, 16, 24, 2)
    n_data = data / normalize_array
    data = n_data.squeeze().tolist()

    six_exercise[check].append(data)


# cos 법칙으로 각 ABC를 °(도) 단위로 구하는 함수
def cal_angle(A, B, C):
    if A == B or C == B:
        return 180
    A, B, C = map(np.array, (A, B, C))
    angle = degrees(arccos(min(max(dot(A - B, C - B) / (norm(B - A) * norm(C - B)), -1.0), 1.0)))
    return angle


# 두 점 사이의 거리를 구하는 함수
def cal_distance(A, B):
    distance = ((A[0] - B[0]) ** 2 + (A[1] - B[1]) ** 2) ** 0.5
    return distance


# 과거와 현재 값 비교해서 얼마나 차이가 나는 지 확인하는 코드
def past_current(past, current, part, error_message: set, message: str, threshold: float, mode=False):
    if past:
        current = part
        if mode:
            if abs(current - past) < threshold:
                error_message.add(message)
        elif abs(current - past) > threshold:
            error_message.add(message)
    return part


# 통계
def mvmm(data):
    if data:
        mean_var_max_min = np.mean(data), np.var(data), max(data), min(data)
        return mean_var_max_min
    else:
        return 0, 0, 0, 0


# 시퀀스마다
def burpees(data):
    result = {'waist': [], 'chest': [], 'arm': [], 'back': [], 'neutral': [], 'shoulder': [], 'hand': []}

    for sequence in data:
        chest_position = []
        waist_position = []
        arm_angle = []
        back_angle = []
        neutral = []
        shoulder = []
        hand = []

        for pts in sequence:
            Nose = pts[0]
            Eye_L, Eye_R = pts[1], pts[2]
            Ear_L, Ear_R = pts[3], pts[4]
            Shoulder_L, Shoulder_R = pts[5], pts[6]
            Elbow_L, Elbow_R = pts[7], pts[8]
            Wrist_L, Wrist_R = pts[9], pts[10]
            Hip_L, Hip_R = pts[11], pts[12]
            Knee_L, Knee_R = pts[13], pts[14]
            Ankle_L, Ankle_R = pts[15], pts[16]
            Neck = pts[17]
            Palm_L, Palm_R = pts[18], pts[19]
            Back, Waist = pts[20], pts[21]
            Foot_L, Foot_R = pts[22], pts[23]

            # 어깨-팔꿈치-손목 각도
            Arm_L = cal_angle(Shoulder_L, Elbow_L, Wrist_L)
            Arm_R = cal_angle(Shoulder_R, Elbow_R, Wrist_R)

            waist_position.append(Waist[0])
            arm_angle.append((Arm_L+Arm_R)/2)

            # 엎드렸을 때(손이 무릎 밑으로 갔을 때)
            if Palm_L[1] < Knee_L[1] and Palm_R[1] < Knee_R[1]:
                # 가슴 이동
                chest_position.append(Back[1])

                # 팔을 폈으면 허리를 폈는지 확인(아래 둘 중 하나만? 둘 다?)
                if Arm_L > 150 and Arm_R > 150:
                    # 목-등-허리 각도 확인
                    back_angle.append(cal_angle(Neck, Back, Waist))

                    # 등-힙 직선과 허리의 높이 비교
                    dx = Back[0] - (Hip_L[0] + Hip_R[0]) / 2
                    if dx:
                        neutral.append(int(Waist[1] <
                                           ((Back[1] - (Hip_L[1] + Hip_R[1]) / 2) / dx)
                                           * (Waist[0] - Back[0]) + Back[1]))
                    else:
                        neutral.append(int(Waist[1] < Back[1]))

                # 팔꿈치와 어깨의 y좌표 비교
                elif Arm_L < 95 and Arm_R < 95:
                    shoulder.append((abs(Shoulder_L[1] - Elbow_L[1]) + abs(Shoulder_R[1] - Elbow_R[1]))/2)

                hand.append(abs(Back[0] - (Palm_L[0] + Palm_R[0]) / 2))

        result['waist'].append(mvmm(waist_position))
        result['chest'].append(mvmm(chest_position))
        result['arm'].append(mvmm(arm_angle))
        result['back'].append(mvmm(back_angle))
        result['neutral'].append(mvmm(neutral))
        result['shoulder'].append(mvmm(shoulder))
        result['hand'].append(mvmm(hand))

    return result

# 푸쉬업 (버피와 거의 같음)
# 정면 / 측면 각도 달라져야 함
def push_up(data):
    result = {'waist': [], 'chest': [], 'arm': [], 'back': [], 'neutral': [], 'shoulder': [], 'hand': []}

    for sequence in data:
        chest_position = []
        waist_position = []
        arm_angle = []
        back_angle = []
        neutral = []
        shoulder = []
        hand = []

        for pts in sequence:
            Nose = pts[0]
            Eye_L, Eye_R = pts[1], pts[2]
            Ear_L, Ear_R = pts[3], pts[4]
            Shoulder_L, Shoulder_R = pts[5], pts[6]
            Elbow_L, Elbow_R = pts[7], pts[8]
            Wrist_L, Wrist_R = pts[9], pts[10]
            Hip_L, Hip_R = pts[11], pts[12]
            Knee_L, Knee_R = pts[13], pts[14]
            Ankle_L, Ankle_R = pts[15], pts[16]
            Neck = pts[17]
            Palm_L, Palm_R = pts[18], pts[19]
            Back, Waist = pts[20], pts[21]
            Foot_L, Foot_R = pts[22], pts[23]

            # 어깨-팔꿈치-손목 각도
            Arm_L = cal_angle(Shoulder_L, Elbow_L, Wrist_L)
            Arm_R = cal_angle(Shoulder_R, Elbow_R, Wrist_R)

            waist_position.append(Waist[0])
            arm_angle.append((Arm_L+Arm_R)/2)

            # 엎드렸을 때(손이 무릎 밑으로 갔을 때)
            if Palm_L[1] < Knee_L[1] and Palm_R[1] < Knee_R[1]:
                # 가슴 이동
                chest_position.append(Back[1])

                # 팔을 폈으면 허리를 폈는지 확인(아래 둘 중 하나만? 둘 다?)
                if Arm_L > 120 and Arm_R > 120:
                    # 목-등-허리 각도 확인
                    back_angle.append(cal_angle(Neck, Back, Waist))

                    # 등-힙 직선과 허리의 높이 비교
                    neutral.append(int(Waist[1] < (Back[1] - (Hip_L[1] + Hip_R[1]) / 2)
                                       / (Back[0] - (Hip_L[0] + Hip_R[0]) / 2) * (Waist[0] - Back[0]) + Back[1]))

                # 팔꿈치와 어깨의 y좌표 비교
                elif Arm_L < 95 and Arm_R < 95:
                    shoulder.append((abs(Shoulder_L[1] - Elbow_L[1]) + abs(Shoulder_R[1] - Elbow_R[1]))/2)

                hand.append(abs(Back[0] - (Palm_L[0] + Palm_R[0]) / 2))

        result['waist'].append(mvmm(waist_position))
        result['chest'].append(mvmm(chest_position))
        result['arm'].append(mvmm(arm_angle))
        result['back'].append(mvmm(back_angle))
        result['neutral'].append(mvmm(neutral))
        result['shoulder'].append(mvmm(shoulder))
        result['hand'].append(mvmm(hand))

    return result

# 사이드 레터럴 레이즈
def side_lateral_raise(data):
    result = {'elbow': [], 'leg': [], 'arm': [], 'top': [], 'shoulder': [], 'hand': []}

    for sequence in data:
        elbow = []
        leg = []  # 발목-무릎-힙 간의 거리: 다리 길이
        top = []  # 목-허리 거리: 상체 길이
        shoulder = []
        arm = []  # 어깨-팔꿈치-손목 각도
        hand = []  # 팔꿈치-손목-손바닥 각도

        for pts in sequence:
            Nose = pts[0]
            Eye_L, Eye_R = pts[1], pts[2]
            Ear_L, Ear_R = pts[3], pts[4]
            Shoulder_L, Shoulder_R = pts[5], pts[6]
            Elbow_L, Elbow_R = pts[7], pts[8]
            Wrist_L, Wrist_R = pts[9], pts[10]
            Hip_L, Hip_R = pts[11], pts[12]
            Knee_L, Knee_R = pts[13], pts[14]
            Ankle_L, Ankle_R = pts[15], pts[16]
            Neck = pts[17]
            Palm_L, Palm_R = pts[18], pts[19]
            Back, Waist = pts[20], pts[21]
            Foot_L, Foot_R = pts[22], pts[23]

            elbow.append(int(Elbow_L[1] > Wrist_L[1] and Elbow_R[1] > Wrist_R[1]))

            # 힙-무릎-발목 거리 평균(왼/오) / 목-허리 높이 기록
            leg.append((cal_distance(Hip_L, Knee_L) + cal_distance(Knee_L, Ankle_L)
                        + cal_distance(Hip_R, Knee_R) + cal_distance(Knee_R, Ankle_R)) / 2)
            top.append(cal_distance(Neck, Waist))

            # 어깨 평균(왼/오) 높이 기록
            shoulder.append((Shoulder_L[1] + Shoulder_R[1]) / 2)

            # 어깨-팔꿈치-손목 / 팔꿈치-손목-손바닥 각도 기록
            arm.append((cal_angle(Shoulder_L, Elbow_L, Wrist_L) + cal_angle(Shoulder_R, Elbow_R, Wrist_R))/2)
            hand.append((cal_angle(Elbow_L, Wrist_L, Palm_L) + cal_angle(Elbow_R, Wrist_R, Palm_R))/2)

        result['elbow'].append(mvmm(elbow))
        result['leg'].append(mvmm(leg))
        result['arm'].append(mvmm(arm))
        result['top'].append(mvmm(top))
        result['shoulder'].append(mvmm(shoulder))
        result['hand'].append(mvmm(hand))

    return result


# 풀업
def pull_up(data):
    result = {'waist': [], 'eye_nose': [], 'ear': [], 'elbow': [], 'shoulder': []}

    for sequence in data:
        waist = []
        elbow = []
        eye_nose = []
        ear = []
        shoulder = []
        Elbow_dist = 0  # 팔꿈치 너비

        for pts in sequence:
            Nose = pts[0]
            Eye_L, Eye_R = pts[1], pts[2]
            Ear_L, Ear_R = pts[3], pts[4]
            Shoulder_L, Shoulder_R = pts[5], pts[6]
            Elbow_L, Elbow_R = pts[7], pts[8]
            Wrist_L, Wrist_R = pts[9], pts[10]
            Hip_L, Hip_R = pts[11], pts[12]
            Knee_L, Knee_R = pts[13], pts[14]
            Ankle_L, Ankle_R = pts[15], pts[16]
            Neck = pts[17]
            Palm_L, Palm_R = pts[18], pts[19]
            Back, Waist = pts[20], pts[21]
            Foot_L, Foot_R = pts[22], pts[23]

            # 눈-코의 중점과 귀의 중점의 y값(높이) 비교
            eye_nose.append(((Eye_L[1] + Eye_R[1]) / 2 + Nose[1]) / 2)
            ear.append((Ear_L[1] + Ear_R[1]) / 2)
            shoulder.append(int((Shoulder_L[1] + Shoulder_R[1]) / 2 < Neck[1]))

            if Shoulder_L[1] > Elbow_L[1] or Shoulder_R[1] > Elbow_R[1]:
                if Elbow_dist == 0:
                    pass
                elbow.append(int(cal_distance(Elbow_L, Elbow_R) > Elbow_dist))
            else:
                Elbow_dist = max(Elbow_dist, cal_distance(Elbow_L, Elbow_R))

            waist.append(Waist[0])

        result['elbow'].append(mvmm(elbow))
        result['waist'].append(mvmm(waist))
        result['eye_nose'].append(mvmm(eye_nose))
        result['ear'].append(mvmm(ear))
        result['shoulder'].append(mvmm(shoulder))

    return result


# 크로스 런지
def cross_lunge(data):
    result = {'eye_nose': [], 'ear': [], 'knee': [], 'knee_foot': [], 'shoulder_back': []}

    for sequence in data:
        eye_nose = []
        ear = []
        knee = []
        knee_foot = []
        shoulder_back = []

        Knee_mL = float("inf")
        Knee_mR = float("inf")

        for pts in sequence:
            Nose = pts[0]
            Eye_L, Eye_R = pts[1], pts[2]
            Ear_L, Ear_R = pts[3], pts[4]
            Shoulder_L, Shoulder_R = pts[5], pts[6]
            Elbow_L, Elbow_R = pts[7], pts[8]
            Wrist_L, Wrist_R = pts[9], pts[10]
            Hip_L, Hip_R = pts[11], pts[12]
            Knee_L, Knee_R = pts[13], pts[14]
            Ankle_L, Ankle_R = pts[15], pts[16]
            Neck = pts[17]
            Palm_L, Palm_R = pts[18], pts[19]
            Back, Waist = pts[20], pts[21]
            Foot_L, Foot_R = pts[22], pts[23]

            # 구부린 무릎의 최소 각의 크기 저장
            if Knee_L[1] > Knee_R[1]:  # 왼발이 앞
                Knee_mL = min(Knee_mL, cal_angle(Hip_L, Knee_L, Ankle_L))
                knee.append(Knee_mL)
                knee_foot.append(abs(Foot_L[0] - Knee_L[0]))

            else:
                Knee_mR = min(Knee_mR, cal_angle(Hip_R, Knee_R, Ankle_R))
                knee.append(Knee_mR)
                knee_foot.append(abs(Foot_R[0] - Knee_R[0]))

            shoulder_back.append(int(Back[0] - (Shoulder_L[0] + Shoulder_R[0])))
            eye_nose.append(((Eye_L[1] + Eye_R[1]) / 2 + Nose[1]) / 2)
            ear.append((Ear_L[1] + Ear_R[1]) / 2)

        result['knee'].append(mvmm(knee))
        result['knee_foot'].append(mvmm(knee_foot))
        result['eye_nose'].append(mvmm(eye_nose))
        result['ear'].append(mvmm(ear))
        result['shoulder_back'].append(mvmm(shoulder_back))

    return result


# 바벨 스쿼트
def barbell_squat(data):
    result = {'eye_nose': [], 'ear': [], 'shoulder_back': [], 'knee': [], 'knee_foot': []}

    for sequence in data:
        shoulder_back = []
        eye_nose = []
        ear = []
        knee = []
        knee_foot = []

        Knee_mL = float("inf")
        Knee_mR = float("inf")

        for pts in sequence:
            Nose = pts[0]
            Eye_L, Eye_R = pts[1], pts[2]
            Ear_L, Ear_R = pts[3], pts[4]
            Shoulder_L, Shoulder_R = pts[5], pts[6]
            Elbow_L, Elbow_R = pts[7], pts[8]
            Wrist_L, Wrist_R = pts[9], pts[10]
            Hip_L, Hip_R = pts[11], pts[12]
            Knee_L, Knee_R = pts[13], pts[14]
            Ankle_L, Ankle_R = pts[15], pts[16]
            Neck = pts[17]
            Palm_L, Palm_R = pts[18], pts[19]
            Back, Waist = pts[20], pts[21]
            Foot_L, Foot_R = pts[22], pts[23]

            Knee_mL = min(Knee_mL, cal_angle(Hip_L, Knee_L, Ankle_L))
            Knee_mR = min(Knee_mR, cal_angle(Hip_R, Knee_R, Ankle_R))
            knee.append((Knee_mL + Knee_mR)/2)

            eye_nose.append(((Eye_L[1] + Eye_R[1]) / 2 + Nose[1]) / 2)
            ear.append((Ear_L[1] + Ear_R[1]) / 2)

            knee_foot.append((abs(Foot_L[0] - Knee_L[0]) + abs(Foot_R[0] - Knee_R[0])) / 2)
            shoulder_back.append(abs(Back[1] - (Shoulder_L[1] + Shoulder_R[1]) / 2))

            # 발바닥 지면 고정(구현 아직 안함)
            # error_message.add("발바닥을 지면에 고정하세요.")

        result['knee'].append(mvmm(knee))
        result['knee_foot'].append(mvmm(knee_foot))
        result['eye_nose'].append(mvmm(eye_nose))
        result['ear'].append(mvmm(ear))
        result['shoulder_back'].append(mvmm(shoulder_back))

    return result

fns = (burpees, cross_lunge, barbell_squat, side_lateral_raise, push_up, pull_up)
types = ['burpees', 'cross_lunge', 'barbell_squat', 'side_lateral_raise', 'push_up', 'pull_up']
result = []

for fn, data, _type in zip(fns, six_exercise.values(), types):
    # print(_type)
    # mvmms = fn(data)
    # print(np.mean(np.array(list(mvmms.values())), axis=1))

    result.append({_type: {k: {s: t for s, t in zip(('mean', 'var', 'max', 'min'), np.mean(np.array(list(v)), axis=0).tolist())} for k, v in fn(data).items()}})
# print(result)

root = "./kpt_statistics"
os.makedirs(root, exist_ok=True)

dst_path = os.path.join(root, "n_stat.json",)
with open(dst_path, 'w', encoding='utf-8') as f:
    json.dump(result, f, indent=4)

# result = {_type: pd.DataFrame(fn(data)) for fn, data, _type in zip(fns, six_exercise.values(), types)}
#
#
# for _type, df in result.items():
#     df.to_csv(f'./kpt_statistics/{_type}.csv', encoding='utf-8')
