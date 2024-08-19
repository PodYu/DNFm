from ADB import scrcpy_adb
import time

controller = scrcpy_adb.ScreenController(device_ip="192.168.8.4:5555")
time.sleep(2)
controller.click(50, 100)
print("scrcpy_adb")