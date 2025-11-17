from ultralytics import YOLO

# Load a model
# model = YOLO('./runs/detect/train4/weights/best.pt')
model = (YOLO("ultralytics/cfg/models/11/yolo11s-FADC.yaml"))

# Train the moder
model.train(data='yolo-KITTI-data.yaml', workers=0, epochs=300, batch=1, name="YOLOv11",device=0)