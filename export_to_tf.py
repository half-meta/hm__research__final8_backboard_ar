from ultralytics import YOLO
import onnx
import tf2onnx
import tensorflow as tf
import os

# Path to your trained YOLOv8 model
best_model_path = r'C:\Users\will\Downloads\basketball backboard.v1i.yolov11\runs\train\exp\weights\best.pt'

# 1. Export to ONNX
onnx_path = os.path.join(os.path.dirname(best_model_path), 'best.onnx')
model = YOLO(best_model_path)
model.export(format='onnx', dynamic=True,
             simplify=True, imgsz=640, optimize=True)

# 2. Convert ONNX to TensorFlow SavedModel
onnx_model = onnx.load(onnx_path)
tf_model_path = "yolov8_saved_model"
spec = (tf.TensorSpec((1, 3, 640, 640), tf.float32, name="images"),)

# Convert ONNX to TensorFlow SavedModel
model_proto, _ = tf2onnx.convert.from_onnx(
    onnx_model, output_path=tf_model_path)

print(f"TensorFlow SavedModel exported to {tf_model_path}")
print("To convert to TensorFlow.js format, run:")
print("tensorflowjs_converter --input_format=tf_saved_model --output_format=tfjs_graph_model yolov8_saved_model web_model")
