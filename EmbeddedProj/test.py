import cv2
import numpy as np
import pygame
import os
from time import sleep

eyeData = "xml/eyes.xml"
faceData = "xml/face.xml"
DOWNSCALE = 3

# os.putenv('SDL_VIDEODRIVER', 'fbcon')  # frame cache

os.putenv('SDL_VIDEODRV','fbcon')
os.putenv('SDL_FBDEV', '/dev/fb0')
# os.putenv('SDL_MOUSEDRV','dummy')
# os.putenv('SDL_MOUSEDEV','/dev/null')
# os.putenv('DISPLAY','')

# 控制布尔值
add_face_rect = True
add_objects = True
add_eye_rect = True

# 初始化 OpenCV 和 pygame
webcam = cv2.VideoCapture('./face2.mp4')
classifier = cv2.CascadeClassifier(eyeData)
faceClass = cv2.CascadeClassifier(faceData)

# 加载眼镜素材
glasses = cv2.imread('assets/glasses.png', cv2.IMREAD_UNCHANGED)
glasses2 = cv2.imread('assets/glasses2.png', cv2.IMREAD_UNCHANGED)
mode_g = 0
ratio = glasses.shape[1] / glasses.shape[0]

# 初始化 pygame 显示
pygame.init()
screen = pygame.display.set_mode((320, 240))  # 根据 PiTFT 的分辨率设置显示大小
pygame.display.set_caption("Webcam Facial Tracking")

# 主循环
running = True
while running:
    sleep(0.05)
    # 读取视频帧
    rval, frame = webcam.read()
    if not rval:
        break

    # frame = cv2.resize(frame, (320, 240))
    # 检测人脸和眼睛
    minisize = (int(frame.shape[1] / DOWNSCALE), int(frame.shape[0] / DOWNSCALE))
    miniframe = cv2.resize(frame, minisize)
    faces = faceClass.detectMultiScale(miniframe)
    eyes = classifier.detectMultiScale(miniframe)

    if add_eye_rect:
        for eye in eyes:
            x, y, w, h = [v * DOWNSCALE for v in eye]
            pts1 = (x, y + h)
            pts2 = (x + w, y)
            cv2.rectangle(frame, pts1, pts2, color=(0, 255, 0), thickness=3)

            if add_objects:
                h = w / ratio
                y += h / 2
                y, x, w, h = int(y), int(x), int(w), int(h)
                smallglasses = cv2.resize(glasses if mode_g == 0 else glasses2, (w, h))
                bg = frame[y:y + h, x:x + w]
                np.multiply(bg, np.atleast_3d(255 - smallglasses[:, :, 3]) / 255.0, out=bg, casting="unsafe")
                np.add(bg, smallglasses[:, :, :3] * np.atleast_3d(smallglasses[:, :, 3]), out=bg)
                frame[y:y + h, x:x + w] = bg

    if add_face_rect:
        for face in faces:
            x, y, w, h = [v * DOWNSCALE for v in face]
            pts1 = (x, y + h)
            pts2 = (x + w, y)
            cv2.rectangle(frame, pts1, pts2, color=(255, 0, 0), thickness=3)

    # 转换图像格式以在 pygame 中显示
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 转换为 RGB 格式
    frame = np.rot90(frame)  # 如果需要旋转图像
    frame_surface = pygame.surfarray.make_surface(frame)  # 转换为 pygame 表面

    # 在 pygame 显示窗口中显示帧
    screen.blit(frame_surface, (0, 0))
    pygame.display.flip()

    # 检查事件和键盘输入
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    # keys = 3
    # if keys in [27, ord('Q'), ord('q')]: # exit on ESC
    #     break

    keys = pygame.key.get_pressed()
    if keys[pygame.K_ESCAPE]:  # 按 ESC 键退出
        running = False
    if keys[pygame.K_1]:
        add_face_rect = not add_face_rect
    if keys[pygame.K_2]:
        add_eye_rect = not add_eye_rect
    if keys[pygame.K_3]:
        add_objects = not add_objects
    if keys[pygame.K_4]:
        mode_g = 1 - mode_g  # 切换眼镜模式

# 释放资源
pygame.quit()
webcam.release()
