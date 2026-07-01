# PCBA 瑕疵檢測實作

```
!pip install ultralytics
```

```
from IPython import display
display.clear_output()
# prevent ultralytics from tracking your activity
!yolo settings sync=False
import ultralytics
ultralytics.checks()
```

```
from ultralytics import YOLO
from IPython.display import display, Image
```

```
from google.colab import drive
drive.mount('/content/drive')
```

```
train: /content/drive/MyDrive/PCBA/train/images
val:  /content/drive/MyDrive/PCBA/valid/images
test: /content/drive/MyDrive/PCBA/test/images
```

```
model = YOLO('yolov8n.pt')
```

```
results = model.train(data='/content/drive/MyDrive/PCBA/data.yaml', epochs=30, imgsz=640)
```

```
from ultralytics import YOLO
from IPython.display import Image, display
import glob
import os
import random
import cv2
from collections import Counter

weight_list = glob.glob("/content/runs/detect/train/weights/best.pt")

DATASET_DIR = "/content/drive/MyDrive/PCBA"

candidate_image_dirs = [
    f"{DATASET_DIR}/test/images",
    f"{DATASET_DIR}/valid/images",
    f"{DATASET_DIR}/val/images",
    f"{DATASET_DIR}/train/images",
]

image_list = []

for img_dir in candidate_image_dirs:
    image_list += glob.glob(f"{img_dir}/*.jpg")
    image_list += glob.glob(f"{img_dir}/*.jpeg")
    image_list += glob.glob(f"{img_dir}/*.png")
    image_list += glob.glob(f"{img_dir}/*.bmp")

if len(image_list) == 0:
    raise FileNotFoundError("找不到資料集圖片，請確認 DATASET_DIR 路徑是否正確")

print(f"找到 {len(image_list)} 張圖片")

sample_num = min(3, len(image_list))
random_images = random.sample(image_list, sample_num)

print("本次隨機選到：")
for img_path in random_images:
    print(img_path)

if len(weight_list) == 0:
    raise FileNotFoundError("找不到 best.pt，請確認前面已經訓練完成")

best_weight = max(weight_list, key=os.path.getmtime)
print("使用權重：", best_weight)

model = YOLO(best_weight)

output_dir = "/content/random_predict_results"
os.makedirs(output_dir, exist_ok=True)

results = model.predict(
    source=random_images,
    conf=0.25,
    imgsz=640,
    iou=0.45,
    save=False
)

for idx, r in enumerate(results):
    image_path = random_images[idx]
    image_name = os.path.basename(image_path)

    total_count = len(r.boxes)

    class_ids = []
    for box in r.boxes:
        class_ids.append(int(box.cls[0]))

    class_counter = Counter(class_ids)

    print("\n==============================")
    print(f"圖片：{image_name}")
    print(f"總共辨識到：{total_count} 個物件")

    for cls_id, count in class_counter.items():
        class_name = model.names[cls_id]
        print(f"{class_name}: {count} 個")
    print("==============================")

    annotated_img = r.plot()

    output_path = os.path.join(output_dir, f"result_{idx+1}_{image_name}")
    cv2.imwrite(output_path, annotated_img)

    display(Image(filename=output_path))
```