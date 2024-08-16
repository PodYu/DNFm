import scrcpy
import cv2

def handle_frame(frame):
    """
    Handles each frame from the screen stream.

    :param frame: A BGR numpy ndarray representing a frame.
    """
    if frame is not None:
        # frame is an bgr numpy ndarray (cv2' default format)
        cv2.imshow("iphone1screen", frame)
    cv2.waitKey(10)
    return frame

def click(x, y):
    """
    Clicks at the specified coordinates on the screen.

    :param x: The x-coordinate of the click location.
    :param y: The y-coordinate of the click location.
    """
    scrcpy.Client(device="192.168.8.4:5555").device.shell(f'input tap {x} {y}')

# 如果你已经知道设备的序列号或 IP 地址
client = scrcpy.Client(
    device="192.168.8.4:5555",
    max_width=1000,
    bitrate=8000000,
    max_fps=30,
    flip=False,
    block_frame=False,
    connection_timeout=3000
)

client.add_listener(scrcpy.EVENT_FRAME, handle_frame)
client.start(threaded=True)

# 示例：模拟点击屏幕
# click(100, 200)  # 在 x=100, y=200 的位置点击
