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

for json_src in jsons_list:
    flag = False
    json_name = json_src.split("\\")[-1].split('.')[0]
    check = json_name.split('-')[-1]
    if check not in six_exercise:
        continue

    with open(json_src, 'r', encoding='utf-8') as f:
        annotations = json.load(f)

    data = []
    frames = annotations['frames']
    type_num = int(annotations['type'])
    for idx, frame in enumerate(frames):
        view = frame[cam]
        img_name = view['img_key'].split('/')[-1]
        pts = view['pts']
        kpt = [list(keypoint.values()) for keypoint in pts.values()]
        _part_list = list(pts.keys())
        if part_list != _part_list:
            flag = True
            ordered_pts = {part: pts[part] for part in part_list}
            annotations['frames'][idx][cam]['pts'] = ordered_pts

    if flag:
        with open(json_src, 'w', encoding='utf-8') as f:
            json.dump(annotations, f, indent=4)
