from ultralytics import YOLO
import cv2

# 加载模型
model = YOLO("yolov8n.pt")  # load a pretrained model

# 打开视频文件或摄像头
video_path = "C:/Users/86139/Desktop/20240403 威海/无人机/DJI_0196.MP4"  # 或者使用 0 代表默认摄像头
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # 使用模型进行预测
    results = model(frame, verbose=False)

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
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 添加类别标签和置信度
            label = f"{model.names[cls]} {conf:.2f}"
            t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
            c2 = x1 + t_size[0], y1 - t_size[1] - 3
            cv2.rectangle(frame, (x1, y1), c2, [0, 0, 255], -1, cv2.LINE_AA)  # filled
            cv2.putText(frame, label, (x1, y1 - 2), 0, 1, [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)

    # 显示图像
    cv2.imshow('Prediction', frame)

    # 按 'q' 键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源并关闭窗口
cap.release()
cv2.destroyAllWindows()
