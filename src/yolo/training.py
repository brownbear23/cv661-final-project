from ultralytics import YOLO

if __name__ == "__main__":
    base_models = ["../library/litter-detection/runs/detect/train/yolov8m_100epochs/weights/best.pt",
                   "../library/YOLO_weights/yolo11m.pt"]


    # Load the model
    model = YOLO(base_models[0])

    # Train the model
    model.train(
        data="./TACO.yaml",
        epochs=100,
        patience=25,
        imgsz=640,
        device='cpu',
        name="new_train",
        pretrained=False,
        optimizer='SGD',
    )

    # Validate the model
    metrics = model.val()