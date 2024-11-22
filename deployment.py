import io
from typing import List, Tuple
import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import onnxruntime as ort
import numpy as np
import onnxruntime as ort
import cv2

class Detection:
    def __init__(self, model_path: str, classes: List[str]):
        self.model_path = model_path
        self.classes = classes
        self.model = self.__load_model()

    def __load_model(self) -> ort.InferenceSession:
        # Load the ONNX model using ONNX Runtime
        session = ort.InferenceSession(self.model_path)

        # Attempt to use GPU, fall back to CPU if unavailable
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        available_providers = session.get_providers()

        if 'CUDAExecutionProvider' not in available_providers:
            print("CUDAExecutionProvider not available, using CPUExecutionProvider.")
            providers = ['CPUExecutionProvider']

        session.set_providers(providers)

        return session

    def __extract_output(self, preds: np.ndarray, image_shape: Tuple[int, int], input_shape: Tuple[int, int], score: float = 0.1, nms: float = 0.0, confidence: float = 0.0) -> dict:
        class_ids, confs, boxes = list(), list(), list()

        image_height, image_width = image_shape
        input_height, input_width = input_shape
        x_factor = image_width / input_width
        y_factor = image_height / input_height
    
        rows = preds.shape[0]
        for i in range(rows):
            row = preds[i]
            conf = row[4]
            classes_score = row[5:]  # All class scores for this prediction

        # Find the index of the class with the highest score
            _, _, _, max_idx = cv2.minMaxLoc(classes_score)
            class_id = max_idx[1]

        # Extract the individual class score for the selected class
            class_score = classes_score[class_id]  # This should now be a scalar value

        # Check if the class score for the chosen class is above the threshold
            if class_score > score:  # Compare with scalar class_score, not an array
                confs.append(conf)
                label = self.classes[int(class_id)]
                class_ids.append(label)
            
            # Extract boxes
                x, y, w, h = row[0], row[1], row[2], row[3]
                left = int((x - 0.5 * w) * x_factor)
                top = int((y - 0.5 * h) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                box = np.array([left, top, width, height])
                boxes.append(box)

        r_class_ids, r_confs, r_boxes = list(), list(), list()
        indexes = cv2.dnn.NMSBoxes(boxes, confs, confidence, nms)  # Non-Maximum Suppression (NMS)
        for i in indexes:
            r_class_ids.append(class_ids[i])
            r_confs.append(confs[i] * 100)  # Multiply confidence by 100 to make it a percentage
            r_boxes.append(boxes[i].tolist())  # Convert box coordinates to a list

        return {
            'boxes': r_boxes,
            'confidences': r_confs,
         'classes': r_class_ids
        }



    def __call__(self, image: np.ndarray, width: int = 640, height: int = 640, score: float = 0.1, nms: float = 0.0, confidence: float = 0.0) -> dict:
        # Preprocess the image into the correct format for ONNX Runtime
        blob = cv2.dnn.blobFromImage(image, 1/255.0, (width, height), swapRB=True, crop=False)

        # Convert the blob to a format suitable for ONNX (typically float32)
        blob = blob.astype(np.float32)

        # Run the model inference using ONNX Runtime
        input_name = self.model.get_inputs()[0].name  # Get the input name for the model
        preds = self.model.run(None, {input_name: blob})  # Get predictions from the model

        # Post-process the results
        results = self.__extract_output(
            preds=preds[0],  # First element is the output
            image_shape=image.shape[:2],
            input_shape=(height, width),
            score=score,
            nms=nms,
            confidence=confidence
        )
        return results

  
  
  
CLASSES_YOLO = ['Bonnet', 'Bumper', 'Dickey', 'Door', 'Fender', 'Light', 'Windshield']



detection = Detection(
   model_path='model.onnx', 
   classes=CLASSES_YOLO
)


app = FastAPI()
@app.post('/detection')
def post_detection(file: bytes = File(...)):
   image = Image.open(io.BytesIO(file)).convert("RGB")
   image = np.array(image)
   image = image[:,:,::-1].copy()
   results = detection(image)
   return results


   
if __name__ == '__main__':
    uvicorn.run("deployment:app", host="127.0.0.1", port=8080)