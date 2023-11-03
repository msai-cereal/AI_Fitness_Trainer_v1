import os
import glob
import cv2
import torch
import random
import numpy as np
import albumentations as A
from ultralytics import YOLO

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

model = YOLO("./runs/pose/train13/weights/best.pt")

root_path = "./ultralytics/cfg/fitness/valid/images"
# image_path_list = glob.glob(os.path.join(root_path, '*jpg'))

# root_path = "C:/Users/labadmin/Downloads/frame_ex/test0000_1078"
# image_path_list = glob.glob(os.path.join(root_path, '*png'))
# root_path = "C:/Users/labadmin/Downloads/frame_ex/img"
# image_path_list += glob.glob(os.path.join(root_path, '*png'))

# root_path = "C:/Users/labadmin/Downloads/20231026"
# image_path_list = glob.glob(os.path.join(root_path, "*", "*", "*.png"))

# root_path = "C:/Users/labadmin/Downloads/crop"
image_path_list = glob.glob(os.path.join(root_path, '*jpg'))

random.shuffle(image_path_list)
# print(image_path_list)
# exit()

# li = list(map(str, range(713, 729)))

for image_path in image_path_list:
    # num = image_path.split("\\")[-1].split("-")[0]
    # if num not in li:
    #     continue
    with torch.no_grad():
        result = model.predict(image_path, save=False, imgsz=640, conf=0.5, device='cuda')[0]

    image_path = result.path
    boxes = result.boxes.xyxy
    cls = result.boxes.cls
    conf = result.boxes.conf
    cls_dict = result.names
    kps = result.keypoints.xy[0].cpu().numpy()
    k_conf = result.keypoints.conf
    if k_conf is None:
        continue
    # print(f"키 포인트 confidence: {k_conf}")

    image = cv2.imread(image_path)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, c = image.shape
    scale_factor_x = 640 / w
    scale_factor_y = 640 / h

    image = cv2.resize(image, (640, 640))

    for box, cls_number, conf in zip(boxes, cls, conf):
        # conf_number = float(conf)
        cls_number_int = int(cls_number)
        cls_name = cls_dict[cls_number_int]
        x1, y1, x2, y2 = box
        x1_int = int(x1.item())
        y1_int = int(y1.item())
        x2_int = int(x2.item())
        y2_int = int(y2.item())

        x1_scale = int(x1_int * scale_factor_x)
        y1_scale = int(y1_int * scale_factor_y)
        x2_scale = int(x2_int * scale_factor_x)
        y2_scale = int(y2_int * scale_factor_y)

        image = cv2.rectangle(image, (x1_scale, y1_scale), (x2_scale, y2_scale), (0, 255, 0), 2)
        cv2.putText(image, cls_name, (x1_scale, y1_scale), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    for i, keypoint in enumerate(kps):
        x, y = int(keypoint[0] * scale_factor_x), int(keypoint[1] * scale_factor_y)
        cv2.circle(image, (x, y), 2, kpt_color[i].tolist(), 2)

    for j, (s, t) in enumerate(skeleton):
        try:
            x1, y1 = int(kps[s][0] * scale_factor_x), int(kps[s][1] * scale_factor_y)
            x2, y2 = int(kps[t][0] * scale_factor_x), int(kps[t][1] * scale_factor_y)
            cv2.line(image, (x1, y1), (x2, y2), limb_color[j].tolist(), 2)
        except IndexError as e:
            print(image_path, j)

    image = cv2.resize(image, (640, 640))
    transform = A.Compose([A.augmentations.crops.transforms.CenterCrop(600, 360, always_apply=False, p=1.0)])
    image = transform(image=image)['image']

    # cv2.imwrite("./test.jpg", image)
    cv2.imshow("Test", image)
    k = cv2.waitKey(0)
    if k == ord('q'):
        break


# 이미지 한 장
# import cv2
# from ultralytics import YOLO

# model = YOLO("./runs/pose/train2/weights/best.pt")

# results = model.predict(
#     "./ultralytics/cfg/fitness/test/images/파일이름.jpg",
#     save=False, imgsz=640, conf=0.5, device='cuda'
# )

# results = model.predict(
#     "./ultralytics/cfg/fitness/train/images/001-1-1-01-Z17_B-0000011.jpg",
#     save=False, imgsz=640, conf=0.5, device='cuda'
# )

# results = model.predict(
#     "bus.jpg",
#     save=False, imgsz=640, conf=0.5, device='cuda'
# )

# for r in results:
#     image_path = r.path
#     boxes = r.boxes.xyxy
#     cls = r.boxes.cls
#     conf = r.boxes.conf
#     cls_dict = r.names
#     kps = r.keypoints.xy[0]
#     print(f"키포인트 confidence: {r.keypoints.conf}")

#     image = cv2.imread(image_path)
#     # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     h, w, c = image.shape
#     scale_factor_x = 640 / w
#     scale_factor_y = 640 / h

#     image = cv2.resize(image, (640, 640))

#     for box, cls_number, conf in zip(boxes, cls, conf):
#         conf_number = float(conf.item())
#         cls_number_int = int(cls_number.item())
#         cls_name = cls_dict[cls_number_int]
#         x1, y1, x2, y2 = box
#         x1_int = int(x1.item())
#         y1_int = int(y1.item())
#         x2_int = int(x2.item())
#         y2_int = int(y2.item())

        
#         x1_scale = int(x1_int * scale_factor_x)
#         y1_scale = int(y1_int * scale_factor_y)
#         x2_scale = int(x2_int * scale_factor_x)
#         y2_scale = int(y2_int * scale_factor_y)

#         image = cv2.rectangle(image, (x1_scale, y1_scale), (x2_scale, y2_scale), (0, 255, 0), 2)

#     for keypoint in kps:
#         x, y = int(keypoint[0].item() * scale_factor_x), int(keypoint[1].item() * scale_factor_y)
#         cv2.circle(image, (x, y), 2, (0, 255, 255), 2)

#     # cv2.imwrite("./test.jpg", image)
#     cv2.imshow("Test", image)
#     cv2.waitKey(0)
#     k = cv2.waitKey(0)
#     if k == ord('q'):
#         break
