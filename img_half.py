import os
import glob
import shutil
from tqdm import tqdm

for mode in ('train', 'valid'):
    # AI hub 데이터 중 json 파일에서 keypoints가 누락된 이미지 제거
    root_path = f"ultralytics/cfg/fitness/{mode}/images"
    imgs_no_label = open(f'C:/Users/labadmin/Desktop/img_{mode}.txt', 'r', encoding='utf-8')
    for img_no_label in tqdm(imgs_no_label):
        img_path = os.path.join(root_path, img_no_label.rstrip())
        os.remove(img_path)

    # 대각선 두 방향 중 한 방향 파일 제거(이동)
    dst_path = f"ultralytics/cfg/fitness/{mode}/half"
    os.makedirs(dst_path, exist_ok=True)
    images_path = f"./ultralytics/cfg/fitness/{mode}/images"
    labels_path = f"./ultralytics/cfg/fitness/{mode}/labels"
    images_list = glob.glob(os.path.join(images_path, "*.jpg"))
    labels_list = glob.glob(os.path.join(labels_path, "*.txt"))

    for img_src in tqdm(images_list):
        img_name = img_src.split("\\")[-1]
        # 풀업 외 운동은 뒷 모습 제외
        if 'D-' in img_name or 'E-' in img_name:
            img_dst = os.path.join(dst_path, img_name)
            shutil.move(img_src, img_dst)

    for txt_src in tqdm(labels_list):
        txt_name = txt_src.split("\\")[-1]
        # 풀업 외 운동은 뒷 모습 제외
        if 'D-' in txt_name or 'E-' in txt_name:
            txt_dst = os.path.join(dst_path, txt_name)
            shutil.move(txt_src, txt_dst)
