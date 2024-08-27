import scrcpy
import cv2
import time
import random

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
    controller = ScreenController(device_ip="192.168.3.43:5555")
    # 示例：模拟点击屏幕
    time.sleep(2)
    print("click")
    controller.click(50, 100)  # 在 x=540, y=1080 的位置点击
