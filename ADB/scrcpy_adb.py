import scrcpy
import cv2
import time
import random
from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)


class ScreenController:
    def __init__(self, device_ip, max_width=1000, bitrate=8000000, max_fps=30, flip=False, block_frame=False, connection_timeout=3000):
        """
        Initializes the ScreenController with the given parameters.

        :param device_ip: The IP address of the device.
        :param max_width: Maximum width of the screen stream.
        :param bitrate: Bitrate of the screen stream.
        :param max_fps: Maximum frames per second of the screen stream.
        :param flip: Whether to flip the screen stream.
        :param block_frame: Whether to block on frame.
        :param connection_timeout: Connection timeout for the device.
        """
        self.client = scrcpy.Client(
            device=device_ip,
            max_width=max_width,
            bitrate=bitrate,
            max_fps=max_fps,
            flip=flip,
            block_frame=block_frame,
            connection_timeout=connection_timeout
        )
        self.client.add_listener(scrcpy.EVENT_FRAME, self.handle_frame)
        self.client.start(threaded=True)

    '''
    def handle_frame(self, frame):
        """
        Handles each frame from the screen stream.

        :param frame: A BGR numpy ndarray representing a frame.
        """
        if frame is not None:
            # frame is an bgr numpy ndarray (cv2's default format)
            cv2.imshow("iphone1screen", frame)
        cv2.waitKey(10)
        return frame
    '''

    def handle_frame(self, frame):
        """
        Handles each frame from the screen stream by first processing it with YOLOv8 and then displaying it.

        :param frame: A BGR numpy ndarray representing a frame.
        """
        if frame is not None and (frame.shape[0] > 0 and frame.shape[1] > 0):
            # 使用模型进行预测
            #resized_frame = cv2.resize(frame, (640, 640))  # Resize frame if needed
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
            # 显示处理后的帧
            cv2.imshow('Prediction', frame)
            cv2.waitKey(10) # Wait for a short period to display the frame
        #else:
            #print("Received an empty or invalid frame.")

    def click(self, x: int, y: int):
        """
        Clicks at the specified coordinates on the screen.

        :param x: The x-coordinate of the click location.
        :param y: The y-coordinate of the click location.
        """
        #client.device.shell(f'input tap {x} {y}')
        print("click_down")
        self.client.control.touch(x, y, action=scrcpy.ACTION_DOWN, touch_id=-1)
        print("click_down")
        time.sleep(random.uniform(0.15, 0.25) )
        self.client.control.touch(x, y, action=scrcpy.ACTION_UP, touch_id=-1)
        print("click_up")

    def move_start(self, start_x, start_y):
        """
        Drags from the start coordinates to the end coordinates on the screen.

        :param start_x: The x-coordinate of the start location.
        :param start_y: The y-coordinate of the start location.
        """
        self.client.control.touch(x, y, action=scrcpy.ACTION_DOWN, touch_id=-1)
        print("click_down")
        time.sleep(random.uniform(0.15, 0.25))
        self.client.control.touch(x, y, action=scrcpy.ACTION_UP, touch_id=-1)
        print("click_up")

# 使用示例
if __name__ == "__main__":
    # 单位电脑连接oneplus
    #controller = ScreenController(device_ip="192.168.8.4:5555")
    # 家里电脑连接mix2
    #controller = ScreenController(device_ip="192.168.3.43:5555")
    # 家里电脑连接1+
    controller = ScreenController(device_ip="192.168.3.103:5555")
    # 示例：模拟点击屏幕
    time.sleep(2)
    print("click")
    #controller.click(50, 100)  # 在 x=540, y=1080 的位置点击
