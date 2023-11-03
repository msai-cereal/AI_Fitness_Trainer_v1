import os
import shutil
from tqdm import tqdm

# src_path = "./ultralytics/cfg/fitness/train/images"
# dst_path = "./ultralytics/cfg/fitness/train/even"
# os.makedirs(dst_path, exist_ok=True)

src_path = "./ultralytics/cfg/fitness/valid/images"
dst_path = "./ultralytics/cfg/fitness/valid/even"
os.makedirs(dst_path, exist_ok=True)

images_list = os.listdir(src_path)

for img_name in tqdm(images_list):
    img_num = int(img_name.split("-")[-1].split(".")[0])
    if img_num % 2:
        continue
    img_src = os.path.join(src_path, img_name)
    img_dst = os.path.join(dst_path, img_name)
    shutil.move(img_src, img_dst)
