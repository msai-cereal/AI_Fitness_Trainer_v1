import os
import glob
import json
from tqdm import tqdm

body_11 = "Day38_201111_F"
body_12 = "Day37_201110_F"
models_11 = ['Z6', 'Z89', 'Z140', 'Z141', 'Z142', 'Z143', 'Z144']
models_12 = ['Z51', 'Z138', 'Z19', 'Z41', 'Z7', 'Z60', 'Z139']
year = {model_n: body_11 for model_n in models_11}
year.update({model_n: body_12 for model_n in models_12})
# push_up에 해당 하는 모델의 img_key(경로) 지정을 위한 딕셔너리

cam = {"A": "view1", "B": "view2", "C": "view3", "D": "view4", "E": "view5"}

part_list = ['Nose', 'Left Eye', 'Right Eye', 'Left Ear', 'Right Ear', 'Left Shoulder', 'Right Shoulder', 'Left Elbow',
             'Right Elbow', 'Left Wrist', 'Right Wrist', 'Left Hip', 'Right Hip', 'Left Knee', 'Right Knee',
             'Left Ankle', 'Right Ankle', 'Neck', 'Left Palm', 'Right Palm', 'Back', 'Waist', 'Left Foot', 'Right Foot']
part_order = {part: idx for idx, part in enumerate(part_list)}

root = 'C:/Users/labadmin/Downloads/cvat_json'  # 경로 수정
path_list = glob.glob(os.path.join(root, '*.json'))
dst_root = './aihub_json'  # 저장할 경로
os.makedirs(dst_root, exist_ok=True)

for json_path in tqdm(path_list):
    file_name = json_path.split("\\")[-1]
    day, m, type_num = file_name.split('-')
    data = {"frames": [], "type": type_num.split(".")[0]}

    with open(json_path, 'r', encoding='utf-8') as f:
        json_file = json.load(f)

    order = {}
    for idx, image_name in enumerate(json_file.keys()):
        image_num = (int(image_name.split(".")[0].split("-")[-1]) - 1) // 2
        order[idx] = image_name

    for idx in range(len(order)):
        anno = json_file[order[idx]]
        img_name = anno["filename"]  # "561-1-3-27-Z138_C-0000001.jpg"
        pts = anno["regions"]
        n = img_name.split("_")[0].split("-")[-1]
        camera = img_name.split("_")[1].split("-")[0]
        key = "/".join((year[n], m, camera))
        img_key = "/".join((key, img_name[:-12], img_name))
        frame = {cam[camera]: {"pts": {}, "active": "Yes", "img_key": img_key}}
        parts = [[] for _ in range(24)]
        for pt in pts:
            region = pt["region_attributes"]
            if "name" in region:
                part = region["name"]
                if part in part_order:
                    x = pt["shape_attributes"]["cx"]
                    y = pt["shape_attributes"]["cy"]
                    parts[part_order[part]] = (x, y, part)
        for x, y, part in parts:
            frame["view3"]["pts"][part] = {"x": x, "y": y}
        data["frames"].append(frame)

    dst_path = os.path.join(dst_root, file_name)
    # JSON 파일로 저장
    with open(dst_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
