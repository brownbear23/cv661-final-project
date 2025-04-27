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
                   "../../library/YOLO_weights/yolo11m.pt",
                   "../../library/YOLO_weights/yolo11m.pt"
                   ]

    data_yamls = ["/workspace/cv661-final-project/src/yolo/data_yaml/m1_yolo8.yaml",
                  "/workspace/cv661-final-project/src/yolo/data_yaml/m2_single.yaml",
                 "/workspace/cv661-final-project/src/yolo/data_yaml/m3_seq_stage1.yaml"]

    train_names = ["m1_yolo8_train_", "m2_single_train_", "m3_seq_stage1_train_"]




    # The first base model should be changed to the most recent m3_seq_stage1_train_*.pt weight
    # base_models = ["/workspace/cv661-final-project/runs/detect/m3_seq_stage1_train_250427_002807/weights/best.pt"
    #               ]
    #
    # data_yamls = ["/workspace/cv661-final-project/src/yolo/data_yaml/m3_seq_stage2.yaml",
    #               ]
    #
    # train_names = ["m3_seq_stage2_train_",
    #                ]


    for base_model, data_yaml, train_name in zip(base_models, data_yamls, train_names):
        # print(base_model, data_yaml, train_name)
        model = YOLO(base_model)
        model.train(
            data=data_yaml,
            epochs=100,
            workers=8,
            patience=25,
            imgsz=640,
            device='0',
            name=train_name+current_time_str,
            pretrained=True,
            optimizer='SGD',
         )

    # Validate the model (however, this uses the latest epoch) - DELETE
    # model.val(
    #     split="test",
    #     name="test_"+current_time_str,
    # )

    end_time = time.time()
    print(f"Script finished in {end_time - start_time:.2f} seconds at [train_{current_time_str}]")
