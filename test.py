# print('hi')
import os
import glob
import json
from tqdm import tqdm

json_list = glob.glob(os.path.join("C:/Users/labadmin/Downloads/fitness/train/labels", '*', '*', '*.json'))


for json_path in tqdm(json_list):
    with open(json_path, 'r', encoding='utf-8') as f:
        anno = json.load(f)
    json_name = json_path.split("\\")[-1]
    if json_name.split('.')[0][-2:] == '3d':
        continue
    
    frames = anno['frames']
    for frame in frames:
        try:
            view = frame['view3']
            img_name = view['img_key']
            if '049-1-1-03-Z4_B' in img_name:
                print(json_path)
                break
        except:
            pass
    