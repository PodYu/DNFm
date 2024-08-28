from ADB import scrcpy_adb
import scrcpy

#controller = scrcpy_adb.ScreenController(device_ip="192.168.3.43:5555")#MIX
controller1 = scrcpy_adb.ScreenController(device_ip="192.168.8.4:5555")#ONE+

controller1.client.add_listener(scrcpy.EVENT_FRAME, yolo_frame)
#controller.click(50, 100)
print("scrcpy_adb")