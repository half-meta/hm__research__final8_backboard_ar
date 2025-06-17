import random
import os
from ultralytics import YOLO
from glob import glob

# Paths
DATA_YAML = 'data.yaml'
TRAIN_DIR = 'train/images'
VALID_DIR = 'valid/images'
VALID_LABELS_DIR = 'valid/labels'

# Train YOLOv8 model (use a valid pretrained model)
model = YOLO('yolov8n.pt')  # Use YOLOv8 nano architecture
model.train(data=DATA_YAML, epochs=50, imgsz=640,
            project='runs/train', name='exp')

# Load best model after training
best_model_path = 'runs/train/exp/weights/best.pt'
model = YOLO(best_model_path)

# Pick a random image from valid set
valid_images = glob(os.path.join(VALID_DIR, '*.jpg'))
if not valid_images:
    raise RuntimeError('No validation images found!')
random_image = random.choice(valid_images)

# Get corresponding label file
image_name = os.path.basename(random_image)
label_file = os.path.join(
    VALID_LABELS_DIR, os.path.splitext(image_name)[0] + '.txt')
if not os.path.exists(label_file):
    raise RuntimeError(f'Label file not found for {random_image}')

# Run inference
results = model(random_image)

# Print results
print(f"Image: {random_image}")
print("Predictions:")
for box in results[0].boxes:
    print(box.xyxy, box.conf, box.cls)

print("\nGround Truth:")
with open(label_file, 'r') as f:
    print(f.read())
