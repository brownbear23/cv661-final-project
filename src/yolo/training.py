from ultralytics import YOLO

if __name__ == "__main__":
    # Define parameters directly
    base_model_name = "../library/litter-detection/runs/detect/train/yolov8m_100epochs/weights/best.pt" #
    num_epochs = 100

    # Load the model
    model = YOLO(base_model_name)

    # Train the model
    model.train(
        data="./TACO.yaml",
        epochs=num_epochs,
        patience=25,
        imgsz=640,
        device='cpu',
        name="new_train",
        pretrained=False,
        optimizer='SGD',
    )

    # Validate the model
    metrics = model.val()