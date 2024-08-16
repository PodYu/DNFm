import scrcpy
from adbutils import adb
import cv2

class ExtendedClient(scrcpy.Client):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def click(self, x, y):
        """
        Clicks at the specified coordinates on the screen.

        :param x: The x-coordinate of the click location.
        :param y: The y-coordinate of the click location.
        """
        self.device.shell(f'input tap {x} {y}')

# 如果你已经知道设备的序列号或 IP 地址
client = ExtendedClient(
    device="192.168.8.4:5555",
    max_width=1000,
    bitrate=8000000,
    max_fps=30,
    flip=False,
    block_frame=False,
    connection_timeout=3000
)

def on_frame(frame):
    if frame is not None:
        # frame is an bgr numpy ndarray (cv2' default format)
        cv2.imshow("iphone1screen", frame)
    cv2.waitKey(10)
    return frame

client.add_listener(scrcpy.EVENT_FRAME, on_frame)
client.start(threaded=True)
