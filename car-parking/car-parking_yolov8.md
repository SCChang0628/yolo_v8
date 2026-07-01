# 車輛偵測專案實作

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
train: /content/drive/MyDrive/car_parking/train/images
val:  /content/drive/MyDrive/car_parking/valid/images
test: /content/drive/MyDrive/car_parking/test/images
```

```
model = YOLO('yolov8n.pt')
```

```
results = model.train(
    data='/content/drive/MyDrive/car_parking/data.yaml',
    epochs=30,
    imgsz=960,
    batch=8,
    patience=30,
    workers=2,
    cache=True
)
```

```
from ultralytics import YOLO
from google.colab import files
from IPython.display import Image, display
import cv2
import glob
import os

weight_list = glob.glob('/content/runs/detect/train-1/weights/best.pt')

if len(weight_list) == 0:
    raise FileNotFoundError("找不到 best.pt，請確認前面已經訓練完成")

best_weight = max(weight_list, key=os.path.getmtime)
print("使用權重：", best_weight)

model = YOLO(best_weight)

uploaded = files.upload()
image_path = list(uploaded.keys())[0]

print("辨識圖片：", image_path)


results = model.predict(
    source=image_path,
    conf=0.30,
    imgsz=960,
    iou=0.45,
    max_det=300,
    augment=False,
    verbose=True
)

r = results[0]

img = cv2.imread(image_path)
h, w = img.shape[:2]
img_area = h * w

filtered_boxes = []

for box in r.boxes:
    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
    conf = float(box.conf[0])
    cls_id = int(box.cls[0])

    box_w = x2 - x1
    box_h = y2 - y1
    box_area = box_w * box_h

    area_ratio = box_area / img_area
    aspect_ratio = box_w / box_h if box_h > 0 else 0

    if area_ratio < 0.0008:
        continue

    if area_ratio > 0.08:
        continue

    if aspect_ratio < 0.25 or aspect_ratio > 4.0:
        continue

    filtered_boxes.append((x1, y1, x2, y2, conf, cls_id))


car_count_raw = len(r.boxes)
car_count_filtered = len(filtered_boxes)

print("==============================")
print(f"YOLO 辨識到：{car_count_filtered} 台車")
print("==============================")


for i, (x1, y1, x2, y2, conf, cls_id) in enumerate(filtered_boxes):
    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

    label = f"car {conf:.2f}"

    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.putText(
        img,
        label,
        (x1, max(y1 - 5, 15)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 0, 0),
        2
    )

output_path = "/content/car_detection_result_filtered.jpg"
cv2.imwrite(output_path, img)

display(Image(filename=output_path))
```
