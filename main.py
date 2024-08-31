from ADB import scrcpy_adb_1
import scrcpy
from YOLO import yolo3_test_video
import cv2

#controller1 = scrcpy_adb.ScreenController(device_ip="192.168.3.43:5555")#MIX
controller1 = scrcpy_adb_1.ScreenController(device_ip="192.168.8.4:5555")  #ONE+
#controller1 = scrcpy_adb_1.ScreenController(device_ip="192.168.3.103:5555")  #ONE+home


while True:
    frame1 = controller1.listener_frame(frame)
    cv2.imshow('Frame', frame1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print('Done')

