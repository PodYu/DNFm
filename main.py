from ADB import scrcpy_adb as kongzhi
import time

kongzhi.click(500,1000)
time.sleep(2)
kongzhi.drag(100, 400, 300, 400)
print("scrcpy_adb")