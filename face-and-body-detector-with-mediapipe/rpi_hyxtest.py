# from picamera2 import Picamera2
# import cv2
# import mediapipe as mp

# # 初始化Mediapipe
# mp_drawing = mp.solutions.drawing_utils
# mp_holistic = mp.solutions.holistic

# # 初始化Picamera2
# picam2 = Picamera2()
# picam2.configure(picam2.create_preview_configuration(main={"size": (320, 240)}))
# picam2.start()

# # 帧计数器
# frame_count = 0
# frame_interval = 5  # 每隔5帧检测和输出一次

# try:
#     with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
#         while True:
#             # 捕获帧
#             frame = picam2.capture_array()

#             if frame is None:
#                 print("无法捕获到画面，请检查相机是否正常运行。")
#                 break

#             # 检查是否需要处理当前帧
#             if frame_count % frame_interval == 0:
#                 # 转换颜色空间为RGB
#                 image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                 image.flags.writeable = False

#                 # Mediapipe处理
#                 results = holistic.process(image)

#                 # 转换回BGR以便渲染
#                 image.flags.writeable = True
#                 image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

#                 # 绘制面部关键点
#                 mp_drawing.draw_landmarks(
#                     image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
#                     mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
#                     mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
#                 )

#                 # 绘制右手关键点
#                 mp_drawing.draw_landmarks(
#                     image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
#                     mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
#                     mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
#                 )

#                 # 绘制左手关键点
#                 mp_drawing.draw_landmarks(
#                     image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
#                     mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
#                     mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
#                 )

#                 # 绘制姿态关键点
#                 mp_drawing.draw_landmarks(
#                     image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
#                     mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
#                     mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
#                 )

#                 # 仅每5帧显示一次检测结果
#                 cv2.imshow("Mediapipe Holistic on PiCamera", image)

#             # 增加帧计数器
#             frame_count += 1

#             # 按 'q' 键退出
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break

# finally:
#     picam2.close()
#     cv2.destroyAllWindows()








# from picamera2 import Picamera2
# import cv2
# import mediapipe as mp
# import os

# # 初始化Mediapipe
# mp_drawing = mp.solutions.drawing_utils
# mp_holistic = mp.solutions.holistic

# # 初始化Picamera2
# picam2 = Picamera2()
# picam2.configure(picam2.create_preview_configuration(main={"size": (320, 240)}))
# picam2.start()

# # # 设置环境变量以将窗口渲染到 PiTFT
# # os.environ["SDL_FBDEV"] = "/dev/fb1"  # 如果是 PiTFT 2.8"，通常使用 fb1
# # os.environ["SDL_VIDEODRIVER"] = "dummy"  # 防止窗口自动显示到 HDMI 屏幕
# # set SDL environment，ensure PiTFT display
# os.putenv('SDL_VIDEODRIVER', 'fbcon')  # frame cache

# os.putenv('SDL_VIDEODRV','fbcon')
# os.putenv('SDL_FBDEV', '/dev/fb1')
# os.putenv('SDL_MOUSEDRV','dummy')
# os.putenv('SDL_MOUSEDEV','/dev/null')
# os.putenv('DISPLAY','')

# # 帧计数器
# frame_count = 0
# frame_interval = 5  # 每隔5帧检测和输出一次

# try:
#     with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
#         while True:
#             # 捕获帧
#             frame = picam2.capture_array()

#             if frame is None:
#                 print("无法捕获到画面，请检查相机是否正常运行。")
#                 break

#             # 检查是否需要处理当前帧
#             if frame_count % frame_interval == 0:
#                 # 转换颜色空间为RGB
#                 image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                 image.flags.writeable = False

#                 # Mediapipe处理
#                 results = holistic.process(image)

#                 # 转换回BGR以便渲染
#                 image.flags.writeable = True
#                 image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

#                 # 绘制面部关键点
#                 mp_drawing.draw_landmarks(
#                     image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
#                     mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
#                     mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
#                 )

#                 # 绘制右手关键点
#                 mp_drawing.draw_landmarks(
#                     image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
#                     mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
#                     mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
#                 )

#                 # 绘制左手关键点
#                 mp_drawing.draw_landmarks(
#                     image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
#                     mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
#                     mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
#                 )

#                 # 绘制姿态关键点
#                 mp_drawing.draw_landmarks(
#                     image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
#                     mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
#                     mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
#                 )

#                 # 创建窗口并将其显示在 PiTFT 屏幕上
#                 cv2.namedWindow("Mediapipe Holistic on PiTFT", cv2.WND_PROP_FULLSCREEN)
#                 cv2.moveWindow("Mediapipe Holistic on PiTFT", 0, 0)  # 窗口移动到 PiTFT 起始位置
#                 cv2.setWindowProperty("Mediapipe Holistic on PiTFT", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
#                 cv2.imshow("Mediapipe Holistic on PiTFT", image)

#             # 增加帧计数器
#             frame_count += 1

#             # 按 'q' 键退出
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break

# finally:
#     picam2.close()
#     cv2.destroyAllWindows()









from picamera2 import Picamera2
import cv2
import mediapipe as mp
import pygame
import numpy as np
import os

# 初始化Pygame并配置显示到PiTFT
# os.environ["SDL_FBDEV"] = "/dev/fb1"  # PiTFT的帧缓冲设备
# os.environ["SDL_VIDEODRIVER"] = "dummy"  # 防止输出到默认显示器
# # set SDL environment，ensure PiTFT display
os.putenv('SDL_VIDEODRIVER', 'fbcon')  # frame cache

os.putenv('SDL_VIDEODRV','fbcon')
os.putenv('SDL_FBDEV', '/dev/fb1')
os.putenv('SDL_MOUSEDRV','dummy')
os.putenv('SDL_MOUSEDEV','/dev/null')
os.putenv('DISPLAY','')


pygame.init()
screen = pygame.display.set_mode((320, 240))  # 根据PiTFT分辨率调整
pygame.mouse.set_visible(False)
clock = pygame.time.Clock()

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

                # 将图像转换为Pygame表面
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换为RGB格式
                image = np.rot90(image)  # 如果显示方向不对，可旋转图像
                surface = pygame.surfarray.make_surface(image)

                # 渲染到PiTFT屏幕
                screen.blit(surface, (0, 0))
                pygame.display.update()

            # 增加帧计数器
            frame_count += 1

            # 检查退出事件
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt
                if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                    raise KeyboardInterrupt

            clock.tick(30)  # 控制帧率

finally:
    picam2.close()
    pygame.quit()
