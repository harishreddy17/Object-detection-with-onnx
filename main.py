import torch

model_path = 'yolov5s.pt'  # Replace with your .pt file
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)  # Load YOLO model
model.eval()

# Export to ONNX
dummy_input = torch.zeros(1, 3, 640, 640)  # Update input dimensions as per your training
onnx_path = model_path.replace('.pt', '.onnx')
torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    opset_version=12,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)
print(f"Converted to ONNX: {onnx_path}")
