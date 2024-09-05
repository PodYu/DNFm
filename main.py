from ADB import scrcpy_adb_1
import time
from ultralytics import YOLO
import cv2
import queue
import threading
import traceback


# 定义一个自动清理队列的类，继承自queue.Queue
class AutoCleaningQueue(queue.Queue):
    # 重写put方法，当队列满时自动丢弃最旧的元素
    def put(self, item, block=True, timeout=None):
        if self.full():
            self.get()  # 自动丢弃最旧的元素
        super().put(item, block, timeout)


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
        self.yolo = YOLO("yolov8n.pt")
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
            self.show_queue.put(result[0].plot())


# 显示推理结果的函数
def show_result(screen):
    """
    显示屏幕图像。

    参数:
    - screen: 需要显示的屏幕图像
    """
    try:
        if screen is not None:
            show_qrt = 1.3
            width = int(896 * show_qrt)
            height = int(414 * show_qrt)
            cv2.namedWindow('screen', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('screen', width, height)
            cv2.imshow('screen', screen)
            cv2.waitKey(1)
    except Exception as e:
        print(f'出现异常:{e}')
        traceback.print_exc()


# 主程序入口
if __name__ == '__main__':
    # 初始化各种队列
    image_queue = AutoCleaningQueue(maxsize=3)
    infer_queue = AutoCleaningQueue(maxsize=3)
    show_queue = AutoCleaningQueue(maxsize=3)
    # controller1 = scrcpy_adb.ScreenController(device_ip="192.168.3.43:5555")#MIX
    controller1 = scrcpy_adb_1.ScreenController(image_queue, device_ip="192.168.8.4:5555")  # ONE+
    # controller1 = scrcpy_adb_1.ScreenController(device_ip="192.168.3.103:5555")  #ONE+home
    yolo = ProcessYolo(image_queue, infer_queue, show_queue)

    # 初始化游戏控制和屏幕处理对象
    control = GameControl(client)
    screen = Screen(image_queue, control)
    action = GameAction(control, infer_queue, screen)
    print("初始化完成，开始处理 主线程")

    # 主循环
    while True:
        if show_queue.empty():
            time.sleep(0.01)
            continue
        image = show_queue.get()
        image_w = 1200
        image = cv2.resize(image, (image_w, int(image.shape[0] * image_w / image.shape[1])))

        # 创建按钮区域并绘制按钮
        button_panel_width = 100
        button_panel = np.zeros((image.shape[0], button_panel_width, 3), dtype=np.uint8)
        button_height = 50
        button_gap = 10
        button_color = (0, 255, 0)  # 绿色按钮
        buttons = ["run", "stop", "reset", "Hongyan", "Naima", "Qigong", "Jianshen", "Bwj"]

        def draw_buttons(panel):
            for i, label in enumerate(buttons):
                y1 = i * (button_height + button_gap) + button_gap
                y2 = y1 + button_height
                cv2.rectangle(panel, (10, y1), (button_panel_width - 10, y2), button_color, -1)
                cv2.putText(panel, label, (20, y1 + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        def on_mouse(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                if x > image.shape[1]:
                    x_in_panel = x - image.shape[1]
                    for i, label in enumerate(buttons):
                        y1 = i * (button_height + button_gap) + button_gap
                        y2 = y1 + button_height
                        if y1 <= y <= y2:
                            handle_button_click(i)
                else:
                    control.click(x / image.shape[1] * 2400, y / image.shape[0] * 1080)

        def handle_button_click(button_index):
            if button_index == 0:
                action.stop_event = False
            elif button_index == 1:
                action.stop_event = True
            elif button_index == 2:
                action.reset()
            elif button_index == 3:
                action.mainVo.role = Role.HONGYAN
                action.param = GameParamVO()
            elif button_index == 4:
                action.mainVo.role = Role.NAIMA
                action.param = GameParamVO()
            elif button_index == 5:
                action.mainVo.role = Role.QIGONG
                action.param = GameParamVO()
            elif button_index == 6:
                action.mainVo.role = Role.Jianshen
                action.param = GameParamVO()
            elif button_index == 7:
                action.param = GameParamVO()
                action.start_bwj()

        def update_display():
            combined = np.hstack((image, button_panel))
            cv2.imshow("Image", combined)

        draw_buttons(button_panel)
        cv2.namedWindow("Image")
        cv2.setMouseCallback("Image", on_mouse)
        update_display()
        cv2.waitKey(1)


