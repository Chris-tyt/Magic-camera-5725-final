# hand test

# from picamera2 import Picamera2
# import cv2
# import mediapipe as mp
# import pygame
# import numpy as np
# import os

# # 配置PiTFT显示
# os.putenv('SDL_VIDEODRIVER', 'fbcon')
# os.putenv('SDL_FBDEV', '/dev/fb1')
# os.putenv('SDL_MOUSEDRV', 'dummy')
# os.putenv('SDL_MOUSEDEV', '/dev/null')

# pygame.init()
# screen = pygame.display.set_mode((320, 240))  # PiTFT分辨率
# pygame.mouse.set_visible(False)
# clock = pygame.time.Clock()

# # 初始化MediaPipe手部模块
# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils

# # 初始化Picamera2
# picam2 = Picamera2()
# picam2.configure(picam2.create_preview_configuration(main={"size": (320, 240)}))
# picam2.start()

# def detect_gesture(hand_landmarks):
#     thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
#     index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
#     middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
#     ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
#     pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
#     index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]

#     # 定义手势规则
#     if index_tip.y < index_mcp.y and middle_tip.y > index_mcp.y and ring_tip.y > index_mcp.y:
#         return "Gesture 1"
#     elif index_tip.y < index_mcp.y and middle_tip.y < index_mcp.y and ring_tip.y > index_mcp.y:
#         return "Gesture 2"
#     elif index_tip.y < index_mcp.y and middle_tip.y < index_mcp.y and ring_tip.y < index_mcp.y:
#         return "Gesture 3"
#     else:
#         return "Unknown Gesture"

# # 帧计数器
# frame_count = 0
# frame_interval = 5  # 每隔5帧处理一次

# try:
#     with mp_hands.Hands(
#         static_image_mode=False,
#         max_num_hands=1,
#         min_detection_confidence=0.5,
#         min_tracking_confidence=0.5) as hands:
        
#         while True:
#             # 捕获帧
#             frame = picam2.capture_array()

#             if frame is None:
#                 print("无法捕获到画面，请检查相机是否正常运行。")
#                 break

#             if frame_count % frame_interval == 0:
#                 # 转换为RGB
#                 image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                 image.flags.writeable = False

#                 # 处理图像以获取手部关键点
#                 results = hands.process(image)

#                 # 转换回BGR以便渲染
#                 image.flags.writeable = True
#                 image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

#                 if results.multi_hand_landmarks:
#                     for hand_landmarks in results.multi_hand_landmarks:
#                         # 绘制关键点
#                         mp_drawing.draw_landmarks(
#                             image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

#                         # 检测手势并输出到终端
#                         gesture = detect_gesture(hand_landmarks)
#                         print(f"Detected Gesture: {gesture}")

#                         # 在图像上显示手势
#                         cv2.putText(image, gesture, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#                 # 转换为Pygame表面
#                 image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#                 image = np.rot90(image)  # 如果显示方向不对，可旋转图像
#                 surface = pygame.surfarray.make_surface(image)

#                 # 显示到PiTFT屏幕
#                 screen.blit(surface, (0, 0))
#                 pygame.display.update()

#             frame_count += 1

#             # 事件处理
#             for event in pygame.event.get():
#                 if event.type == pygame.QUIT:
#                     raise KeyboardInterrupt
#                 if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
#                     raise KeyboardInterrupt

#             clock.tick(30)  # 控制帧率

# finally:
#     picam2.close()
#     pygame.quit()


# hand and pose test

# from picamera2 import Picamera2
# import cv2
# import mediapipe as mp
# import pygame
# import numpy as np
# import os

# # 配置PiTFT显示
# os.putenv('SDL_VIDEODRIVER', 'fbcon')
# os.putenv('SDL_FBDEV', '/dev/fb1')
# os.putenv('SDL_MOUSEDRV', 'dummy')
# os.putenv('SDL_MOUSEDEV', '/dev/null')

# pygame.init()
# screen = pygame.display.set_mode((320, 240))  # PiTFT分辨率
# pygame.mouse.set_visible(False)
# clock = pygame.time.Clock()

# # 初始化Mediapipe
# mp_hands = mp.solutions.hands
# mp_holistic = mp.solutions.holistic
# mp_drawing = mp.solutions.drawing_utils

# # 初始化Picamera2
# picam2 = Picamera2()
# picam2.configure(picam2.create_preview_configuration(main={"size": (320, 240)}))
# picam2.start()

# def detect_gesture(hand_landmarks):
#     """检测手势并返回结果"""
#     thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
#     index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
#     middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
#     ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
#     pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
#     index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]

#     # 定义手势规则
#     if index_tip.y < index_mcp.y and middle_tip.y > index_mcp.y and ring_tip.y > index_mcp.y:
#         return "Gesture 1"
#     elif index_tip.y < index_mcp.y and middle_tip.y < index_mcp.y and ring_tip.y > index_mcp.y:
#         return "Gesture 2"
#     elif index_tip.y < index_mcp.y and middle_tip.y < index_mcp.y and ring_tip.y < index_mcp.y:
#         return "Gesture 3"
#     else:
#         return "Unknown Gesture"

# # 帧计数器
# frame_count = 0
# frame_interval = 5  # 每隔5帧处理一次

# try:
#     with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic, \
#          mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:

#         while True:
#             # 捕获帧
#             frame = picam2.capture_array()
#             if frame is None:
#                 print("无法捕获到画面，请检查相机是否正常运行。")
#                 break

#             # 转换为RGB
#             image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             image.flags.writeable = False

#             # Mediapipe处理
#             face_results = holistic.process(image)  # 人脸和姿态处理
#             hand_results = hands.process(image)    # 手部处理

#             # 转换回BGR以便渲染
#             image.flags.writeable = True
#             image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

#             # 绘制人脸关键点
#             if face_results.face_landmarks:
#                 mp_drawing.draw_landmarks(
#                     image, face_results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
#                     mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
#                     mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
#                 )

#             # 绘制手部关键点并检测手势
#             if hand_results.multi_hand_landmarks:
#                 for hand_landmarks in hand_results.multi_hand_landmarks:
#                     mp_drawing.draw_landmarks(
#                         image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
#                         mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
#                         mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
#                     )

#                     # 检测手势并输出
#                     gesture = detect_gesture(hand_landmarks)
#                     print(f"Detected Gesture: {gesture}")
#                     cv2.putText(image, gesture, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#             # 将图像转换为Pygame表面
#             image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#             image = np.rot90(image)  # 如果显示方向不对，可旋转图像
#             surface = pygame.surfarray.make_surface(image)

#             # 显示到PiTFT屏幕
#             screen.blit(surface, (0, 0))
#             pygame.display.update()

#             # 帧计数器
#             frame_count += 1

#             # 事件处理
#             for event in pygame.event.get():
#                 if event.type == pygame.QUIT:
#                     raise KeyboardInterrupt
#                 if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
#                     raise KeyboardInterrupt

#             clock.tick(30)  # 控制帧率

# finally:
#     picam2.close()
#     pygame.quit()



# use multiple threads to detect face and hand
from picamera2 import Picamera2
import cv2
import mediapipe as mp
import pygame
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor

# 配置PiTFT显示
os.putenv('SDL_VIDEODRIVER', 'fbcon')
os.putenv('SDL_FBDEV', '/dev/fb1')
os.putenv('SDL_MOUSEDRV', 'dummy')
os.putenv('SDL_MOUSEDEV', '/dev/null')

pygame.init()
screen = pygame.display.set_mode((320, 240))  # PiTFT分辨率
pygame.mouse.set_visible(False)
clock = pygame.time.Clock()

# 初始化Mediapipe模块
mp_hands = mp.solutions.hands
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# 初始化Picamera2
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": (320, 240)}))
picam2.start()

def process_faces_and_pose(image, holistic_model):
    """
    人脸和姿态检测任务
    """
    results = holistic_model.process(image)
    return results

def process_hands(image, hands_model):
    """
    手势检测任务
    """
    results = hands_model.process(image)
    return results

# 帧间隔控制
frame_count = 0
frame_interval = 5  # 每隔 5 帧检测一次

try:
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic_model, \
         mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands_model:

        # 创建线程池
        with ThreadPoolExecutor(max_workers=2) as executor:
            while True:
                # 捕获帧
                frame = picam2.capture_array()

                if frame is None:
                    print("无法捕获到画面，请检查相机是否正常运行。")
                    break

                # 转换为RGB
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False

                # 控制帧间隔
                frame_count += 1
                if frame_count % frame_interval == 0:
                    # 提交任务到线程池
                    face_pose_future = executor.submit(process_faces_and_pose, image, holistic_model)
                    hands_future = executor.submit(process_hands, image, hands_model)

                    # 获取线程结果
                    face_pose_results = face_pose_future.result()
                    hands_results = hands_future.result()

                    # 转换回BGR
                    image.flags.writeable = True
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                    # 绘制人脸关键点
                    if face_pose_results.face_landmarks:
                        mp_drawing.draw_landmarks(
                            image, face_pose_results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                            mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                            mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
                        )

                    # 绘制手部关键点
                    if hands_results.multi_hand_landmarks:
                        for hand_landmarks in hands_results.multi_hand_landmarks:
                            mp_drawing.draw_landmarks(
                                image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
                            )

                            # 检测手势
                            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                            middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                            ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
                            pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
                            index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]

                            gesture = "Unknown Gesture"
                            if index_tip.y < index_mcp.y and middle_tip.y > index_mcp.y and ring_tip.y > index_mcp.y:
                                gesture = "Gesture 1"
                            elif index_tip.y < index_mcp.y and middle_tip.y < index_mcp.y and ring_tip.y > index_mcp.y:
                                gesture = "Gesture 2"
                            elif index_tip.y < index_mcp.y and middle_tip.y < index_mcp.y and ring_tip.y < index_mcp.y:
                                gesture = "Gesture 3"
                            else:
                                gesture = "Unknown Gesture"

                            print(f"Detected Gesture: {gesture}")
                            cv2.putText(image, gesture, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    # 转换为Pygame表面
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image = np.rot90(image)
                    surface = pygame.surfarray.make_surface(image)

                    # 显示到PiTFT屏幕
                    screen.blit(surface, (0, 0))
                    pygame.display.update()

                # 事件处理
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        raise KeyboardInterrupt
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                        raise KeyboardInterrupt

                clock.tick(30)  # 控制帧率

finally:
    picam2.close()
    pygame.quit()