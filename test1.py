import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision.ops import nms

# Define the process for filtering boxes and applying NMS
def process_boxes_and_scores(boxes, scores, score_threshold=0.5, iou_threshold=0.5, class_names=None):
    # Step 1: Filter out low-confidence boxes
    valid_boxes = []
    valid_scores = []
    valid_classes = []
    
    # Iterate through each class's score to filter valid boxes
    for class_idx in range(scores.shape[1]):  # Iterating through classes
        class_scores = scores[0, class_idx].flatten()  # Flatten to get all scores for this class
        class_boxes = boxes[0, class_idx].reshape(-1, 4)  # Reshape boxes for the current class

        # Filter based on score threshold
        valid_indices = class_scores > score_threshold
        valid_boxes_class = class_boxes[valid_indices]
        valid_scores_class = class_scores[valid_indices]
        
        # Append to the valid lists
        valid_boxes.extend(valid_boxes_class)
        valid_scores.extend(valid_scores_class)
        valid_classes.extend([class_idx] * len(valid_scores_class))  # Store class index for each valid score
    
    valid_boxes = np.array(valid_boxes)
    valid_scores = np.array(valid_scores)
    valid_classes = np.array(valid_classes)

    # Step 2: Apply Non-Maximum Suppression (NMS) per class
    selected_boxes = []
    selected_scores = []
    selected_classes = []

    for class_idx in np.unique(valid_classes):  # Apply NMS for each class separately
        class_mask = valid_classes == class_idx
        class_boxes = valid_boxes[class_mask]
        class_scores = valid_scores[class_mask]
        
        # Convert to tensors for NMS
        boxes_tensor = torch.tensor(class_boxes, dtype=torch.float32)
        scores_tensor = torch.tensor(class_scores, dtype=torch.float32)
        
        # Apply NMS
        selected_indices = nms(boxes_tensor, scores_tensor, iou_threshold)
        
        # Get the selected boxes and scores
        selected_boxes_class = class_boxes[selected_indices]
        selected_scores_class = class_scores[selected_indices]
        
        selected_boxes.extend(selected_boxes_class)
        selected_scores.extend(selected_scores_class)
        selected_classes.extend([class_idx] * len(selected_scores_class))
    
    selected_boxes = np.array(selected_boxes)
    selected_scores = np.array(selected_scores)
    selected_classes = np.array(selected_classes)

    # Step 3: Map class indices to class names safely
    selected_class_names = []
    for class_idx in selected_classes:
        if class_idx < len(class_names):  # Check if class_idx is within the bounds
            selected_class_names.append(class_names[class_idx])
        else:
            selected_class_names.append(f"Unknown Class {class_idx}")  # Fallback for unknown classes
    
    return selected_boxes, selected_scores, selected_class_names


# Visualization function to draw bounding boxes on the image
def visualize_boxes(image_path, selected_boxes, selected_class_names):
    # Load image
    image = cv2.imread(image_path)

    # Draw boxes and labels on image
    for box, class_name in zip(selected_boxes, selected_class_names):
        x1, y1, x2, y2 = box
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(image, class_name, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display image
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()


# Main function to load model, process boxes, apply NMS, and visualize results
def main(model_path, image_path, class_names):
    # Load model (replace with actual model loading)
    # model = torch.load(model_path)

    # Placeholder for boxes and scores (replace these with actual model outputs)
    boxes = np.random.rand(1, 71, 80, 80, 4)  # Replace with actual box data
    scores = np.random.rand(1, 71, 80, 80)  # Replace with actual score data

    # Step 1: Process boxes and scores
    selected_boxes, selected_scores, selected_class_names = process_boxes_and_scores(
        boxes, scores, score_threshold=0.5, iou_threshold=0.5, class_names=class_names
    )

    # Step 2: Visualize the results
    visualize_boxes(image_path, selected_boxes, selected_class_names)


# Example usage
if __name__ == "__main__":
    model_path = "best.onnx"  # Update with your model path
    image_path = "car2.jpg"  # Update with your image path
    
    # Define class names (make sure this list corresponds to your model's classes)
    class_names = [
        'class1', 'class2', 'class3', 'class4', 'class5', 'class6', 'class7', 'class8', 
        'class9', 'class10', 'class11', 'class12', 'class13', 'class14', 'class15', 
        'class16', 'class17', 'class18', 'class19', 'class20', 'class21', 'class22', 
        'class23', 'class24', 'class25', 'class26', 'class27', 'class28', 'class29', 
        'class30', 'class31', 'class32', 'class33', 'class34', 'class35', 'class36', 
        'class37', 'class38', 'class39', 'class40', 'class41', 'class42', 'class43', 
        'class44', 'class45', 'class46', 'class47', 'class48', 'class49', 'class50', 
        'class51', 'class52', 'class53', 'class54', 'class55', 'class56', 'class57', 
        'class58', 'class59', 'class60', 'class61', 'class62', 'class63', 'class64', 
        'class65', 'class66', 'class67', 'class68', 'class69', 'class70', 'class71'
    ]

    # Call the main function
    main(model_path, image_path, class_names)
