import time
from picamera2 import Picamera2
import cv2
import mediapipe as mp
import pygame
import numpy as np
import os
import imutils

# 初始化Pygame并配置显示到PiTFT
os.putenv('SDL_VIDEODRIVER', 'fbcon')  # frame cache
os.putenv('SDL_VIDEODRV','fbcon')
os.putenv('SDL_FBDEV', '/dev/fb1')
os.putenv('SDL_MOUSEDRV','dummy')
os.putenv('SDL_MOUSEDEV','/dev/null')
os.putenv('DISPLAY','')

pygame.init()
screen = pygame.display.set_mode((320, 240))
pygame.mouse.set_visible(False)
clock = pygame.time.Clock()

# 初始化Mediapipe和Picamera2
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": (320, 240)}))
picam2.start()

# 读取眼镜和帽子图片
# 请替换为实际的图片路径
glasses_img = cv2.imread('assets/glasses.png', cv2.IMREAD_UNCHANGED)  # 包含透明通道
hat_img = cv2.imread('assets/hat.png', cv2.IMREAD_UNCHANGED)  # 包含透明通道

frame_count = 0
frame_interval = 5

def overlay_image(bg_image, overlay_image, x, y, overlay_size):
    overlay = cv2.resize(overlay_image, overlay_size)
    h, w, _ = overlay.shape
    alpha_overlay = overlay[:, :, 3] / 255.0
    alpha_bg = 1.0 - alpha_overlay

    for c in range(0, 3):
        bg_image[y:y+h, x:x+w, c] = (alpha_overlay * overlay[:, :, c] +
                                     alpha_bg * bg_image[y:y+h, x:x+w, c])

try:
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while True:
            start_total = time.time()  # 测量总时间

            # 捕获帧
            start_capture = time.time()
            frame = picam2.capture_array()
            # print(f"Capture time: {time.time() - start_capture:.6f} s")

            if frame is None:
                print("无法捕获到画面，请检查相机是否正常运行。")
                break

            if frame_count % frame_interval == 0:
                # 转换为RGB
                start_convert = time.time()
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # print(f"Color conversion time: {time.time() - start_convert:.6f} s")

                # Mediapipe处理
                start_mediapipe = time.time()
                results = holistic.process(image)
                # print(f"Mediapipe process time: {time.time() - start_mediapipe:.6f} s")

                # 绘制关键点
                # start_draw = time.time()
                # mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
                # mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                # mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                # mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
                # print(f"Drawing landmarks time: {time.time() - start_draw:.6f} s")

                # 添加眼镜和帽子
                if results.face_landmarks:
                    h, w, _ = image.shape
                    # 计算眼镜位置
                    left_eye = results.face_landmarks.landmark[33]  # 左眼门组中的关键点
                    right_eye = results.face_landmarks.landmark[263]  # 右眼门组中的关键点
                    x1 = int(left_eye.x * w)
                    y1 = int(left_eye.y * h)
                    x2 = int(right_eye.x * w)
                    y2 = int(right_eye.y * h)
                    glasses_width = x2 - x1
                    glasses_height = int(glasses_width * glasses_img.shape[0] / glasses_img.shape[1])
                    
                    # 计算眼镜的旋转角度
                    eye_center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    angle = np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi  # 计算角度
                    glasses_img_rotated = imutils.rotate(glasses_img, angle)  # 旋转眼镜图像
                    
                    # 确保背景区域的大小与旋转后的眼镜图像匹配
                    glasses_height = int(glasses_width * glasses_img_rotated.shape[0] / glasses_img_rotated.shape[1])
                    if y1 - int(glasses_height / 2) < 0:  # 确保不超出图像边界
                        y1 = int(glasses_height / 2)
                    # 确保y1不超出图像上边界
                    if y1 + int(glasses_height / 2) > h:
                        y1 = h - int(glasses_height / 2)
                    overlay_image(image, glasses_img_rotated, x1, y1 - int(glasses_height / 2), (glasses_width, glasses_height))

                    # 计算帽子位置
                    forehead = results.face_landmarks.landmark[10]  # 头顶的关键点
                    hat_width = glasses_width * 2
                    hat_height = int(hat_width * hat_img.shape[0] / hat_img.shape[1])
                    x_hat = int(forehead.x * w) - glasses_width // 2
                    y_hat = int(forehead.y * h) - int(hat_height * 0.75)  # 调整帽子位置
                    
                    # 确保y_hat不超出图像上边界
                    if y_hat < 0:
                        y_hat = 0
                    # 确保y_hat + hat_height不超出图像下边界
                    if y_hat + hat_height > h:
                        y_hat = h - hat_height

                    # 确保帽子图像的大小与计算的位置相匹配
                    hat_resized = cv2.resize(hat_img, (hat_width, hat_height))
                    overlay_image(image, hat_resized, x_hat, y_hat, (hat_width, hat_height))

                # 转换为Pygame表面并显示
                start_display = time.time()
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                surface = pygame.surfarray.make_surface(np.rot90(image))
                screen.blit(surface, (0, 0))
                pygame.display.update()
                # print(f"Display update time: {time.time() - start_display:.6f} s")

            frame_count += 1
            # print(f"Total frame time: {time.time() - start_total:.6f} s\n")

            # 退出检查
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                    raise KeyboardInterrupt

            clock.tick(30)

finally:
    picam2.close()
    pygame.quit()
