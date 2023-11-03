import os
import glob
import shutil
from tqdm import tqdm

images_path = "./ultralytics/cfg/fitness/train/images"
os.makedirs(images_path, exist_ok=True)
# images_path = "./ultralytics/cfg/fitness/valid/images"
# os.makedirs(images_path, exist_ok=True)

root_path = "C:/Users/labadmin/Downloads/fitness/train/images/71315_68"
images_list = glob.glob(os.path.join(root_path, '*/*/*/*', "*.jpg"))
# root_path = "C:/Users/labadmin/Downloads/fitness/valid/images"
# images_list = glob.glob(os.path.join(root_path, '*/*/*/*', "*.jpg"))

for img_src in tqdm(images_list):
    img_name = img_src.split("\\")[-1]
    img_num = int(img_name.split("-")[-1].split(".")[0])
    if img_num % 2 == 0:
        continue
    type_num = int(img_name[0:3])
    # 버피 테스트, 크로스 런지, 바벨 스쿼트, 사이드 레터럴 레이즈, 푸쉬업, 풀업
    if type_num not in list(range(49, 81)) + list(range(177, 185)) \
            + list(range(313, 329)) + list(range(377, 409)) \
            + list(range(561, 592)) + list(range(713, 729)):
        continue
    # if type_num not in list(range(177, 185)) + list(range(313, 329)):
    #     continue
    # 풀업 외 운동은 뒷 모습 제외
    if type_num not in list(range(713, 728)) and 'A-' in img_name or 'E-' in img_name:
        continue
    if type_num in list(range(713, 728)) and 'B-' in img_name or 'D-' in img_name:
        continue
    img_dst = os.path.join(images_path, img_name)
    shutil.move(img_src, img_dst)


# 풀업 대각선 앞 방향 파일 제거(이동)
# dst_path = "C:/Users/labadmin/Downloads/fitness/train/images/138_12_1/Day33_201105_F/B_D"
# images_path = "./ultralytics/cfg/fitness/train/images"
# images_list = glob.glob(os.path.join(images_path, "*.jpg"))
#
# for img_src in tqdm(images_list):
#     img_name = img_src.split("\\")[-1]
#     type_num = int(img_name[0:3])
#     # 풀업 외 운동은 뒷 모습 제외
#     if type_num in list(range(713, 728)) and 'B-' in img_name or 'D-' in img_name:
#         img_dst = os.path.join(dst_path, img_name)
#         shutil.move(img_src, img_dst)
