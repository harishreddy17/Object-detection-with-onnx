import cv2
import numpy as np
import onnxruntime as ort

# Load ONNX model
session = ort.InferenceSession("yolov5s.onnx")

# Function to preprocess a frame
def preprocess_frame(frame):
    input_image = cv2.resize(frame, (640, 640))
    input_image = input_image / 255.0  # Normalize
    input_image = np.transpose(input_image, (2, 0, 1))  # HWC to CHW
    input_image = np.expand_dims(input_image, axis=0).astype(np.float32)  # Add batch dimension
    return input_image

# Function to post-process outputs
def postprocess(outputs, frame, conf_threshold=0.5):
    raw_output = outputs[0]
    detections = raw_output[0]
    height, width, _ = frame.shape
    results = []

    for detection in detections:
        confidence = detection[4]
        if confidence > conf_threshold:
            x_center, y_center, box_width, box_height = detection[:4]
            x_min = int((x_center - box_width / 2) * width / 640)
            y_min = int((y_center - box_height / 2) * height / 640)
            x_max = int((x_center + box_width / 2) * width / 640)
            y_max = int((y_center + box_height / 2) * height / 640)
            class_id = np.argmax(detection[5:])
            results.append({"bbox": [x_min, y_min, x_max, y_max], "class_id": class_id, "confidence": confidence})
    return results

# Function to draw detections on the frame
def draw_detections(frame, results):
    for result in results:
        x_min, y_min, x_max, y_max = result["bbox"]
        class_id = result["class_id"]
        confidence = result["confidence"]
        # Draw bounding box
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
        # Add label
        label = f"Class {class_id}: {confidence:.2f}"
        cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    return frame

# Process video
video_path = "demo.mp4"  # Replace with your video path or use 0 for webcam
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Cannot open video.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break  # End of video

    # Preprocess the frame
    input_image = preprocess_frame(frame)

    # Inference
    outputs = session.run(None, {"input": input_image})

    # Post-process detections
    results = postprocess(outputs, frame)

    # Draw detections
    frame = draw_detections(frame, results)

    # Display the frame
    cv2.imshow("Video Detection", frame)

    # Break on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
