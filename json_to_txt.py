import os
import glob
import json
from tqdm import tqdm

# views = ('view1', 'view2', 'view3', 'view4', 'view5')
views = ('view1', 'view2', 'view3')
image_width = 1920
image_height = 1080

# 6 classes
# class_num_dict = {n: i for i, r in enumerate((range(49, 81), range(177, 185), range(313, 329),
#                                               range(377, 409), range(561, 592), range(713, 729))) for n in r}
# class_num_dict = {n: i for i, r in enumerate((range(177, 185), range(313, 329)), 1) for n in r}  # 런지, 스쿼트 추가

for mode in ('train', 'valid'):
    labels_path = f"./ultralytics/cfg/fitness/{mode}/labels"
    os.makedirs(labels_path, exist_ok=True)

    if mode == 'train':
        li = ['138_12_1', '269_34_2', '71315_68']
    else:
        li = [""]

    for nums in li:
        root_path = f"C:/Users/labadmin/Downloads/fitness/{mode}/labels/{nums}"
        jsons_list = glob.glob(os.path.join(root_path, "*", "*.json"))

        for json_src in tqdm(jsons_list):
            json_name = json_src.split("\\")[-1]
            if json_name.split('.')[0][-2:] == '3d':
                continue

            with open(json_src, 'r', encoding='utf-8') as f:
                annotations = json.load(f)

            frames = annotations['frames']
            type_num = int(annotations['type'])
            # 버피 테스트, 크로스 런지, 바벨 스쿼트, 사이드 레터럴 레이즈, 푸쉬업, 풀업
            if type_num not in list(range(49, 81)) + list(range(177, 185)) \
                    + list(range(313, 329)) + list(range(377, 409)) \
                    + list(range(561, 592)) + list(range(713, 729)):
                continue
            # if type_num not in list(range(177, 185)) + list(range(313, 329)):
            #     continue

            for frame in frames:
                for cam in views:
                    # 풀업 외 운동은 뒷 모습 제외
                    # if type_num not in list(range(713, 728)) and cam == 'view1' or cam == 'view5':
                    #     continue
                    # if type_num in list(range(713, 728)) and cam == 'view2' or cam == 'view4':
                    #     continue
                    if type_num not in list(range(713, 728)) and cam == 'view1':
                        continue
                    if type_num in list(range(713, 728)) and cam == 'view2':
                        continue

                    view = frame[cam]
                    img_name = view['img_key'].split('/')[-1]
                    pts = view['pts']
                    active = view['active']
                    labels_dst = os.path.join(labels_path, img_name.replace('jpg', 'txt'))

                    # 키 포인트 정보를 상대 좌표로 변환, 텍스트 파일로 저장
                    output_text = ''
                    xs = []
                    ys = []
                    for keypoint in pts.values():
                        x, y = keypoint['x'], keypoint['y']
                        xs.append(x)
                        ys.append(y)

                        relative_x = x / image_width
                        relative_y = y / image_height
                        output_text += f' {relative_x} {relative_y} 2'

                    # class_num, center_x, center_y, width, height 추가
                    w = (max(xs) - min(xs)) / image_width
                    h = (max(ys) - min(ys)) / image_height
                    cx = min(xs) / image_width + w/2
                    cy = min(ys) / image_height + h/2
                    # output_text = f'{class_num_dict[type_num]} {cx} {cy} {w} {h}' + output_text
                    output_text = f'0 {cx} {cy} {w} {h}' + output_text

                    # 결과를 텍스트 파일로 저장
                    with open(labels_dst, 'w', encoding='utf-8') as f:
                        f.write(output_text)
# txt 파일 제거
# import os
# from tqdm import tqdm
# image_path = "C:/Users/labadmin/Downloads/fitness/train/images/138_12_1/Day33_201105_F/B_D"
# file_list = os.listdir(image_path)
# labels_path = "./ultralytics/cfg/fitness/train/labels"
# for file_name in tqdm(file_list):
#     file_path = os.path.join(labels_path, file_name.replace("jpg", "txt"))
#     if os.path.exists(file_path):
#         os.remove(file_path)
