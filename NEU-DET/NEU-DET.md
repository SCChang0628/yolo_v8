# GitHub 下載專案實作

```
!git clone https://github.com/Marfbin/NEU-DET-with-yolov8.git
```

```
%cd /content/NEU-DET-with-yolov8
```

```
!pip install ultralytics
```

```
from google.colab import drive 
drive.mount('/content/drive')
```

```
import os
```

```
!unzip -q /content/drive/MyDrive/archive.zip -d /content/
```

```
yaml_content = """
train: /content/NEU-DET/train/images
val: /content/NEU-DET/validation/images
```

```
nc: 6
names: ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches’]
"""
```

```
with open('/content/NEU-DET-with-yolov8/data.yaml', 'w') as f:
    f.write(yaml_content)
```

```
import torch
from ultralytics import YOLO
```

```
if not hasattr(torch, '_load_patched'):
    _original_load = torch.load
    def _patched_load(*args, **kwargs):
        kwargs['weights_only'] = False
        return _original_load(*args, **kwargs)
    torch.load = _patched_load
    torch._load_patched = True
```

```
model = YOLO('yolov8n.pt')
```

```
results = model.train(
    data='/content/NEU-DET-with-yolov8/data.yaml', 
    epochs=50, 
    imgsz=640, 
    device=0, 
    project='NEU_Defect_Detection', 
    name='yolov8_base'
)
```

```
import glob
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from ultralytics import YOLO
import random
model = YOLO('/content/NEU-DET-with-yolov8/NEU_Defect_Detection/yolov8_base/weights/best.pt')
test_images = glob.glob('/content/NEU-DET-with-yolov8/data/NEU-DET/test/images/*.jpg')
sample_images = random.sample(test_images, 3)
model.predict(sample_images, save=True, project='Predictions', name='test_results', exist_ok=True)
predicted_images = glob.glob('Predictions/test_results/*.jpg')
predicted_images.sort(key=os.path.getmtime, reverse=True)
predicted_images = predicted_images[:3]
plt.figure(figsize=(15, 5))
for i, img_path in enumerate(predicted_images):
    img = mpimg.imread(img_path)
    plt.subplot(1, 3, i+1)
    plt.imshow(img)
    plt.axis('off')
plt.tight_layout()
plt.show()
```