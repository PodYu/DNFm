from ultralytics import YOLO
import cv2

# 加载图像
image_path = "C:/Users/13920/Desktop/000000007281.jpg"
image = cv2.imread(image_path)

# Load a model
#model = YOLO("yolov8n.yaml")  # build a new model from scratch
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

# Use the model
#model.train(data="coco8.yaml", epochs=3)  # train the model
#metrics = model.val()  # evaluate model performance on the validation set
results = model("C:/Users/13920/Desktop/000000007281.jpg")  # predict on an image
#path = model.export(format="onnx")  # export the model to ONNX format

# 绘制预测结果
for r in results:
    boxes = r.boxes
    for box in boxes:
        # 获取边界框坐标
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # 获取类别和置信度
        cls = int(box.cls[0])
        conf = box.conf[0]

        # 绘制边界框
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 添加类别标签和置信度
        label = f"{model.names[cls]} {conf:.2f}"
        t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
        c2 = x1 + t_size[0], y1 - t_size[1] - 3
        cv2.rectangle(image, (x1, y1), c2, [0, 0, 255], -1, cv2.LINE_AA)  # filled
        cv2.putText(image, label, (x1, y1 - 2), 0, 1, [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)

# 显示图像
cv2.imshow('Prediction', image)
cv2.waitKey(0)
cv2.destroyAllWindows()