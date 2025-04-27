from ultralytics import YOLO
from datetime import datetime
import pytz

# Set up timestamp
est = pytz.timezone('US/Eastern')
current_time = datetime.now(est)
current_time_str = current_time.strftime("%y%m%d_%H%M%S")
print(f"Processing: test_best_{current_time_str}")

NORMAL_TESTING = False

if NORMAL_TESTING:
    print("Normal testing for all three models")
    best_models = ["/workspace/cv661-final-project/runs/detect/m1_yolo8_train_250427_002807/weights/best.pt",
                   "/workspace/cv661-final-project/runs/detect/m2_single_train_250427_011321/weights/best.pt",
                   "/workspace/cv661-final-project/runs/detect/m3_seq_stage2_train_250427_011321/weights/best.pt"]

    test_names = ["m1_yolo8_test_",
                   "m2_single_test_",
                   "m3_seq_stage2_test_"]

    for best_model, test_name in zip(best_models, test_names):
        # Load the best model
        model = YOLO(best_model)

        # Validate the best model
        model.val(
            split='test',  # or split='val' if you want validation set
            name=test_name + current_time_str,
        )

else:
    print("m2_single_HOTACO_only_test")
    # For HOTACO test data set only validation
    test_name = "m2_single_HOTACO_only_test_"
    best_model = "/workspace/cv661-final-project/runs/detect/m2_single_train_250427_011321/weights/best.pt"
    model = YOLO(best_model)
    model.val(
        data="/workspace/cv661-final-project/src/yolo/data_yaml/m2_single_HOTACO_only_test.yaml",
        split='test',
        name=test_name + current_time_str,
    )

