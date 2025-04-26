from ultralytics import YOLO
import cv2
from src.util.drawer.img_drawer import draw_on_image

def run_inference(model_path, source_path, save_predictions, output_path):
    # Load the model
    model = YOLO(model_path)

    # Perform inference
    results = model.predict(source=source_path, save=save_predictions)

    detection = results[0]
    # print(detection.names)

    # Save the result image
    if not save_predictions:
        original_image = detection.orig_img.copy()
        # Draw predictions
        for box in detection.boxes:
            cls = int(box.cls[0])
            cls_name = detection.names[cls]
            score = round(float(box.conf[0]), 2)
            bound = box.xyxy[0].tolist()
            draw_on_image(original_image, cls_name, score, bound)
        cv2.imwrite(output_path, original_image)

# Example usage
if __name__ == "__main__":
    run_inference(
        model_path="/workspace/cv661-final-project/runs/detect/new_train8/weights/best.pt",
        source_path="/workspace/cv661-final-project/src/yolo/0584.jpg",
        save_predictions=True,
        output_path=""
    )
