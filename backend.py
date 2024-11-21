import cv2
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, UploadFile
from io import BytesIO
from fastapi.responses import StreamingResponse
import tempfile
import shutil
import os

app = FastAPI()

# Load ONNX model
session = ort.InferenceSession("yolov5s.onnx")

def process_frame(frame):
    # Prepare the frame for inference
    input_image = cv2.resize(frame, (640, 640))
    input_image = input_image / 255.0  # Normalize
    input_image = np.transpose(input_image, (2, 0, 1))  # HWC to CHW
    input_image = np.expand_dims(input_image, axis=0).astype(np.float32)
    
    # Perform inference
    outputs = session.run(None, {"input": input_image})

    # Process the output (for example, draw bounding boxes, etc.)
    for output in outputs[0][0]:
        if output[4] > 0.5:  # Confidence threshold
            x1, y1, x2, y2 = output[:4].astype(int)
            frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return frame

@app.post("/upload_video/")
async def upload_video(file: UploadFile = File(...)):
    video = await file.read()
    
    # Save the video content to a temporary file
    temp_video_path = tempfile.mktemp(suffix=".mp4")
    with open(temp_video_path, "wb") as f:
        f.write(video)
    
    # Open the video with OpenCV
    cap = cv2.VideoCapture(temp_video_path)
    if not cap.isOpened():
        return {"error": "Error opening video"}

    frame_list = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process each frame
        processed_frame = process_frame(frame)

        # Convert frame to a byte array to send as a response
        _, buffer = cv2.imencode('.jpg', processed_frame)
        frame_list.append(buffer.tobytes())

    cap.release()

    # Clean up the temporary video file
    os.remove(temp_video_path)

    # If no frames were processed, return an error
    if not frame_list:
        return {"error": "No frames were processed"}

    # Create a generator for streaming video frames
    def video_stream():
        for frame in frame_list:
            yield frame

    return StreamingResponse(video_stream(), media_type="video/mp4")
