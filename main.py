from ADB import scrcpy_adb_1
import scrcpy
from YOLO import yolo3_test_video
import cv2

#controller1 = scrcpy_adb.ScreenController(device_ip="192.168.3.43:5555")#MIX
#controller1 = scrcpy_adb.ScreenController(device_ip="192.168.8.4:5555")  #ONE+
controller1 = scrcpy_adb_1.ScreenController(device_ip="192.168.3.103:5555")  #ONE+home

frame1 = yolo3_test_video.detect_objects_in_frame(frame)
# 显示图像
cv2.imshow('Frame', frame1)

# 检查退出条件
if cv2.waitKey(1) & 0xFF == ord('q'):
    return False  # 返回 False 表示停止监听

return True  # 返回 True 表示继续监听


controller1.client.add_listener(scrcpy.EVENT_FRAME, yolo_frame)

print("scrcpy_adb")
