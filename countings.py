import numpy as np
from numpy import degrees, arccos, dot
from numpy.linalg import norm

# 정규화된 키포인트
pt = np.random.uniform
data = np.array([[np.squeeze(np.array([[pt(0, 1), pt(0, 1)] for _ in range(24)])) for _ in range(16)]])
data = data.tolist()

"""
Nose = pts[0]
Eye_L, Eye_R = pts[1], pts[2]
Ear_L, Ear_R = pts[3], pts[4]
Shoulder_L, Shoulder_R = pts[5], pts[6]
Elbow_L, Elbow_R = pts[7], pts[8]
Wrist_L, Wrist_R = pts[9], pts[10]
Hip_L, Hip_R = pts[9], pts[10]
Knee_L, Knee_R = pts[11], pts[12]
Ankle_L, Ankle_R = pts[13], pts[14]
Neck = pts[17]
Palm_L, Palm_R = pts[18], pts[19]
Back, Waist = pts[20], pts[21]
Foot_L, Foot_R = pts[22], pts[23]
"""


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


# 시퀀스마다
def count_burpees(data, flag):
    pts = data[0]
    Shoulder_L, Shoulder_R = pts[5], pts[6]
    Elbow_L, Elbow_R = pts[7], pts[8]
    Wrist_L, Wrist_R = pts[9], pts[10]
    Knee_L, Knee_R = pts[13], pts[14]
    Palm_L, Palm_R = pts[18], pts[19]

    # 어깨-팔꿈치-손목 각도
    Arm_L = cal_angle(Shoulder_L, Elbow_L, Wrist_L)
    Arm_R = cal_angle(Shoulder_R, Elbow_R, Wrist_R)

    # 엎드렸을 때(손이 무릎 밑으로 갔을 때)
    if Palm_L[1] > Knee_L[1] and Palm_R[1] > Knee_R[1]:
        if Arm_L > 150 and Arm_R > 150:
            pass

        # 팔을 구부렸으면, 충분히 구부렸는지 확인(아래 둘 중 하나만? 둘 다?)
        # 어깨-팔꿈치-손목의 각도 확인
        elif not flag and (Arm_L < 50 or Arm_R < 50):
            flag = True
        # 팔꿈치와 어깨의 y좌표 비교
        elif not flag and (abs(Shoulder_L[1] - Elbow_L[1]) < 0.02 or abs(Shoulder_R[1] - Elbow_R[1]) < 0.02):
            flag = True

    elif flag:
        return 1, False

    return 0, flag


# 푸쉬업 (버피와 거의 같음)
# 정면 / 측면 각도 달라져야 함
def count_push_up(data, flag):
    pts = data[0]
    Shoulder_L, Shoulder_R = pts[5], pts[6]
    Elbow_L, Elbow_R = pts[7], pts[8]
    Wrist_L, Wrist_R = pts[9], pts[10]
    Knee_L, Knee_R = pts[11], pts[12]
    Palm_L, Palm_R = pts[18], pts[19]

    # 어깨-팔꿈치-손목 각도
    Arm_L = cal_angle(Shoulder_L, Elbow_L, Wrist_L)
    Arm_R = cal_angle(Shoulder_R, Elbow_R, Wrist_R)

    if (Arm_L + Arm_R) > 240 and flag:
        return 1, False

    # 엎드렸을 때(손이 무릎 밑으로 갔을 때)
    if Palm_L[1] > Knee_L[1] and Palm_R[1] > Knee_R[1]:
        # 팔을 폈으면 허리를 폈는지 확인(아래 둘 중 하나만? 둘 다?)
        if Arm_L > 150 and Arm_R > 150:
            pass

        # 팔을 구부렸으면, 충분히 구부렸는지 확인(아래 둘 중 하나만? 둘 다?)
        # 어깨-팔꿈치-손목의 각도 확인
        elif Arm_L < 95 or Arm_R < 95:
            flag = True
        # 팔꿈치와 어깨의 y좌표 비교
        elif abs(Shoulder_L[1] - Elbow_L[1]) < 0.05 or abs(Shoulder_R[1] - Elbow_R[1]) < 0.05:
            flag = True

    return 0, flag


# 사이드 레터럴 레이즈
def count_side_lateral_raise(data, flag):
    pts = data[0]

    Wrist_L, Wrist_R = pts[9], pts[10]
    Back, Waist = pts[20], pts[21]

    if (Wrist_L[1] + Wrist_R[1]) / 2 < Back[1] + 0.02 and flag:
        return 1, False
    elif (Wrist_L[1] + Wrist_R[1]) / 2 > Waist[1] + 0.01:
        flag = True

    return 0, flag


# 풀업
def count_pull_up(data, flag):
    pts = data[0]
    Nose = pts[0]
    Elbow_L, Elbow_R = pts[7], pts[8]
    Palm_L, Palm_R = pts[18], pts[19]

    if abs(Nose[1] - Palm_L[1]) < 0.05 or abs(Nose[1] - Palm_R[1]) < 0.05 and flag:
        return 1, False

    if abs(Nose[1] - Elbow_L[1]) < 0.07 or abs(Nose[1] - Elbow_R[1]) < 0.07:
        flag = True

    return 0, flag


# 크로스 런지
def count_cross_lunge(data, flag):
    pts = data[0]
    Hip_L, Hip_R = pts[11], pts[12]
    Knee_L, Knee_R = pts[13], pts[14]

    # 구부린 무릎의 최소 각의 크기 저장
    if Knee_L[1] < Knee_R[1]:  # 왼발이 앞
        Knee_dL = cal_distance(Hip_L, Knee_L)
        if Knee_dL > 0.16:
            flag = True
        if Knee_dL < 0.125 and flag:
            return 1, False

    else:
        Knee_dR = cal_distance(Hip_R, Knee_R)
        if Knee_dR > 0.16:
            flag = True
        if Knee_dR < 0.125 and flag:
            return 1, False

    return 0, flag


# 바벨 스쿼트
def count_barbell_squat(data, flag):
    pts = data[0]
    Hip_L, Hip_R = pts[11], pts[12]
    Knee_L, Knee_R = pts[13], pts[14]
    Ankle_L, Ankle_R = pts[15], pts[16]

    Knee_cL = cal_angle(Hip_L, Knee_L, Ankle_L)
    Knee_cR = cal_angle(Hip_R, Knee_R, Ankle_R)

    if (Knee_cL > 160 or Knee_cR > 160) and flag:
        return 1, False
    elif Knee_cL < 130 or Knee_cR < 130:
        flag = True

    return 0, flag
