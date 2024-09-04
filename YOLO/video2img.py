import cv2

# 打开视频文件
cap = cv2.VideoCapture('F:\\dataset\\dnfdateset\\img\\Screenrecorder-2024-08-24-18-40-46-781.mp4')

# 读取视频帧并保存为图片
count = 0
print("video is open")
filecount = 0
while cap.isOpened():

    ret, frame = cap.read()

    if ret:
        if count % 10 == 0:  # 采样频率

            filename = 'F:\\dataset\\dnfdateset\\img\\0824\\01\\082402_ls'+str(filecount)+'.jpg'
            print("write img "+filename)
            filecount += 1
            cv2.imwrite(filename, frame)
        count += 1
    else:
        break

# 关闭视频文件
cap.release()
