# 訓練 YOLOv8實作

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
train: /content/drive/MyDrive/yolov8/train/images
val:  /content/drive/MyDrive/yolov8/valid/images
test: /content/drive/MyDrive/yolov8/test/images
```

```
model = YOLO('yolov8n.pt')
```

```
results = model.train(data='/content/drive/MyDrive/yolov8/data.yaml', epochs=50, imgsz=640)
```

```
from ultralytics import YOLO
# 加載訓練好的模型
model = YOLO("runs/detect/train-2/weights/best.pt")
# 測試單張圖像
results = model.predict(source="/content/yolo_2.jpg", save=True, imgsz=640)
print(results)
```

```
from ultralytics import YOLO
# 加載訓練好的模型
model = YOLO("runs/detect/train-2/weights/best.pt")
# 測試文件夾中的多張圖像
results = model.predict(source="/content", save=True, imgsz=640)
print(results)
```

```
from ultralytics import YOLO
# 加載訓練好的模型
model = YOLO("runs/detect/train-2/weights/best.pt")
results = model.predict(source="/content/yolo_4.mp4", save=True, imgsz=640)
print(results)
```