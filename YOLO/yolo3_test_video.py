from ultralytics import YOLO
import cv2
import threading
import time

# 加载模型
model = YOLO("yolov8n.pt")

# 定义一个处理YOLO模型推理的类
class ProcessYolo():
    def __init__(self, image_queue, infer_queue, show_queue):
        """
        初始化ProcessYolo类。

        参数:
        - image_queue: 图像队列，用于存储待处理的图像
        - infer_queue: 推理结果队列，用于存储模型推理的结果
        - show_queue: 显示队列，用于存储需要显示的图像
        """
        self.image_queue = image_queue
        self.infer_queue = infer_queue
        self.show_queue = show_queue
        self.yolo = model
        self.thread = threading.Thread(target=self.handle)  # 创建线程，并指定目标函数
        self.thread.start()

    def handle(self):
        """
        处理队列中的图像，使用YOLO模型进行推理。
        """
        while True:
            if self.image_queue.empty():
                time.sleep(0.005)
                continue
            result = self.yolo.predict(source=self.image_queue.get(), verbose=False)
            if result is None:
                continue
            self.infer_queue.put(result)
            # 绘制预测结果到show_queue
            for r in result:
                boxes = r.boxes
                for box in boxes:
                    # 获取边界框坐标
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    # 获取类别和置信度
                    cls = int(box.cls[0])
                    conf = box.conf[0]

                    # 绘制边界框
                    cv2.rectangle(results, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # 添加类别标签和置信度
                    label = f"{model.names[cls]} {conf:.2f}"
                    t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
                    c2 = x1 + t_size[0], y1 - t_size[1] - 3
                    cv2.rectangle(result, (x1, y1), c2, [0, 0, 255], -1, cv2.LINE_AA)  # filled
                    cv2.putText(result, label, (x1, y1 - 2), 0, 1, [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)

            self.show_queue.put(result)


def show_objects_in_frame(results):
    """
    使用YOLOv8模型对单个视频帧进行目标检测，并返回带有标注的帧。

    参数:
    frame (numpy.ndarray): 输入的视频帧。

    返回:
    numpy.ndarray: 处理后的带有标注的视频帧。
    """

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
            cv2.rectangle(results, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 添加类别标签和置信度
            label = f"{model.names[cls]} {conf:.2f}"
            t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
            c2 = x1 + t_size[0], y1 - t_size[1] - 3
            cv2.rectangle(results, (x1, y1), c2, [0, 0, 255], -1, cv2.LINE_AA)  # filled
            cv2.putText(results, label, (x1, y1 - 2), 0, 1, [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)

    return results

# 示例调用
if __name__ == "__main__":
    # 打开视频文件或摄像头
    video_path = "C:/Users/86139/Desktop/20240403 威海/无人机/DJI_0196.MP4"
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # 对每个帧进行目标检测
        annotated_frame = detect_objects_in_frame(frame)

        # 显示图像
        cv2.imshow('Prediction', annotated_frame)

        # 按 'q' 键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源并关闭窗口
    cap.release()
    cv2.destroyAllWindows()
