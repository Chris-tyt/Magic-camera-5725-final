from picamera2 import Picamera2
import cv2
import mediapipe as mp
import pygame,pigame
import time
import sys
# import RPi.GPIO as GPIO
import os

# ---------------------------------------- PiTFT -------------------------------------------

# set SDL environment，ensure PiTFT display
os.putenv('SDL_VIDEODRIVER', 'fbcon')  # frame cache

os.putenv('SDL_VIDEODRV','fbcon')
os.putenv('SDL_FBDEV', '/dev/fb1')
os.putenv('SDL_MOUSEDRV','dummy')
os.putenv('SDL_MOUSEDEV','/dev/null')
os.putenv('DISPLAY','')

# Pygame setup
pygame.init()
pitft = pigame.PiTft()
screen_size = (320, 240)
screen = pygame.display.set_mode(screen_size)

small_font = pygame.font.Font(None, 15)
medium_font = pygame.font.Font(None, 20)
large_font = pygame.font.Font(None, 25)

# Colors
RED = (255, 0, 0)
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Display functions
def draw_screen():
    screen.fill(WHITE)

    # Quit button
    text_quit = large_font.render("QUIT", True, BLACK)
    text_quit_rect = text_quit.get_rect(center=(160, 220))  # Set center to (170, 220)
    screen.blit(text_quit, text_quit_rect)

    pygame.display.flip()

# ---------------------------------------- PiTFT -------------------------------------------

# 初始化Mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# 初始化Picamera2
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": (320, 240)}))
picam2.start()

# 帧计数器
frame_count = 0
frame_interval = 5  # 每隔5帧检测和输出一次

try:
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while True:
            # 捕获帧
            frame = picam2.capture_array()
            pitft.update()

            if frame is None:
                print("无法捕获到画面，请检查相机是否正常运行。")
                break

            # 检查是否需要处理当前帧
            if frame_count % frame_interval == 0:
                # 转换颜色空间为RGB
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False

                # Mediapipe处理
                results = holistic.process(image)

                # 转换回BGR以便渲染
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # 绘制面部关键点
                mp_drawing.draw_landmarks(
                    image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                    mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                    mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
                )

                # 绘制右手关键点
                mp_drawing.draw_landmarks(
                    image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
                )

                # 绘制左手关键点
                mp_drawing.draw_landmarks(
                    image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
                )

                # 绘制姿态关键点
                mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                )

                # 仅每5帧显示一次检测结果
                cv2.imshow("Mediapipe Holistic on PiCamera", image)

            # 增加帧计数器
            frame_count += 1

            draw_screen()

            # 按 'q' 键退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

finally:
    picam2.close()
    cv2.destroyAllWindows()





