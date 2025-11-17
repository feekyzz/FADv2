from ultralytics import YOLO

# Load a model
# model = YOLO('./runs/detect/train4/weights/best.pt')
model = (YOLO("ultralytics/cfg/models/BDD/yolo11s.yaml"))
# Train the moder
model.train(data='yolo-BDD100K-data.yaml', workers=0, epochs=300, batch=16, name="BDD100K-4head")
