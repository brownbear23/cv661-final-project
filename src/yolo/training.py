from ultralytics import YOLO
import time
from datetime import datetime
import pytz

# Define EST timezone
est = pytz.timezone('US/Eastern')
current_time = datetime.now(est)
current_time_str = current_time.strftime("%y%m%d_%H%M%S")

if __name__ == "__main__":
    start_time = time.time()
    print(f"Processing: train_{current_time_str}")

    base_models = ["../../library/litter-detection/runs/detect/train/yolov8m_100epochs/weights/best.pt",
                   "../../library/YOLO_weights/yolo11m.pt"]

    # Load the model
    model = YOLO(base_models[0])

    # Train the model
    model.train(
        data="HOTACO.yaml",
        # epochs=100,
        epochs=2,
        patience=25,
        imgsz=640,
        device='0',
        name="train_"+current_time_str,
        pretrained=True,
        optimizer='SGD',
    )

    # Validate the model (however, this uses the latest epoch)
    # model.val(
    #     split="test",
    #     name="test_"+current_time_str,
    # )

    end_time = time.time()
    print(f"Script finished in {end_time - start_time:.2f} seconds at [train_{current_time_str}]")
