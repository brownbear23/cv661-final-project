from ultralytics import YOLO
from datetime import datetime
import pytz

# Set up timestamp
est = pytz.timezone('US/Eastern')
current_time = datetime.now(est)
current_time_str = current_time.strftime("%y%m%d_%H%M%S")
print(f"Processing: test_best_{current_time_str}")

# Load the best model
model = YOLO("/workspace/cv661-final-project/runs/detect/train_250425_222350/weights/best.pt")

# Validate the best model
model.val(
    split='test',  # or split='val' if you want validation set
    name="test_best_" + current_time_str,
)
