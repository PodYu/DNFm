import cv2
import numpy as np
from mss import mss
from ultralytics import YOLO

# 加载模型
model = YOLO("yolov8n.pt")

# 指定屏幕捕获的区域
monitor = {"top": 287, "left": 289, "width": 1222, "height": 687}

sct = mss()

while True:
    # 捕获屏幕区域
    sct_img = sct.grab(monitor)

    # 将捕获的区域转换为 OpenCV 图像
    img = cv2.cvtColor(np.array(sct_img), cv2.COLOR_BGRA2BGR)

    # 使用模型进行预测
    results = model(img, verbose=False)

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
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 添加类别标签和置信度
            label = f"{model.names[cls]} {conf:.2f}"
            t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
            c2 = x1 + t_size[0], y1 - t_size[1] - 3
            cv2.rectangle(img, (x1, y1), c2, [0, 0, 255], -1, cv2.LINE_AA)  # filled
            cv2.putText(img, label, (x1, y1 - 2), 0, 1, [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)

    # 显示图像
    cv2.namedWindow('Prediction', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Prediction', 800, 600)  # 设置窗口宽度为 800，高度为 600
    cv2.imshow('Prediction', img)

    # 按 'q' 键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源并关闭窗口
cv2.destroyAllWindows()
