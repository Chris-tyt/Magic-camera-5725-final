from picamera2 import Picamera2
import cv2
import mediapipe as mp
import pygame
import numpy as np
import os
import sys
import RPi.GPIO as GPIO
import time
import pygame,pigame
from pygame.locals import *

# ----------------- flag set -----------------
RUNNING = True
in_start_menu = True

# ----------------- time set -----------------
start_time = time.time()
run_time = 100
fps_start_time = time.time()
fps = 0

# ----------------- GPIO set -----------------
GPIO.setmode(GPIO.BCM)
GPIO.setup(27, GPIO.IN, pull_up_down=GPIO.PUD_UP)

def GPIO27_callback(channel):
    global RUNNING
    print("Quit....")
    RUNNING = False

GPIO.add_event_detect(27, GPIO.FALLING, callback=GPIO27_callback, bouncetime=200)

# Set SDL environment to ensure PiTFT display
os.putenv('SDL_VIDEODRIVER', 'fbcon')  # Frame cache
os.putenv('SDL_FBDEV', '/dev/fb0')
os.putenv('SDL_MOUSEDRV', 'dummy')
os.putenv('SDL_MOUSEDEV', '/dev/null')
os.putenv('DISPLAY', '')

# ----------------- pygame set -----------------
# Initialize pygame
pygame.init()
pitft = pigame.PiTft()
screen_size = (320, 240)
screen = pygame.display.set_mode(screen_size)
pygame.display.set_caption('Magic Menu')
screen.fill((0,0,0))
pygame.display.update()


# ----------------- display set -----------------
# pygame.mouse.set_visible(False)
# Colors
ORANGE = (255, 165, 0)
YELLOW = (255, 255, 0)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
WHITE = (255, 255, 255)

font = pygame.font.Font(None, 30)
font_mid = pygame.font.Font(None, 45)
font_big = pygame.font.Font(None, 60)
options = ['first', 'second', 'third']
positions = [(140, 120), (140, 150), (140, 180)]
clock = pygame.time.Clock()

# ----------------- camera set -----------------
# Initialize PiCamera2
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": (320, 240)}))
picam2.start()

# Initialize MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
# mp_holistic = mp.solutions.holistic

# ----------------- frame set -----------------
frame_count = 0
frame_interval = 2
result = None

fps_frame_count = 0

# ----------------- mode flags -----------------
mode1 = False
mode2 = False
mode3 = False

# Global variables for countdown
countdown_active = False
countdown_start_time = 0
countdown_duration = 3  # 3 seconds countdown
new_mode = None

# Display start menu with project title and options
def display_start_menu():
    # Fill top half with orange and bottom half with yellow
    screen.fill(ORANGE, rect=(0, 0, 320, 120))
    screen.fill(YELLOW, rect=(0, 120, 320, 120))

    # Display title in the middle of the top half
    title_surface_b = font_big.render('Magic', True, WHITE)
    title_rect_b = title_surface_b.get_rect(center=(170, 50))
    screen.blit(title_surface_b, title_rect_b)

    title_surface = font_big.render('Magic', True, BLACK)
    title_rect = title_surface.get_rect(center=(160, 60))
    screen.blit(title_surface, title_rect)

    # Display options in a single row, evenly spaced
    start_x = 40
    spacing = (320 - 2 * start_x) // (len(options) - 1)
    y_position = 180

    for i, option in enumerate(options):
        option_surface = font.render(option, True, RED)
        option_rect = option_surface.get_rect(center=(start_x + i * spacing, y_position))
        screen.blit(option_surface, option_rect)

    pygame.display.update()

# Function to check which option is selected
def check_option_selection(x, y):
    global mode1, mode2, mode3
    # Detect which button is pressed
    start_x = 40
    spacing = (320 - 2 * start_x) // (len(options) - 1)
    y_position = 180

    for i, option in enumerate(options):
        option_rect = pygame.Rect(start_x + i * spacing - 40, y_position - 20, 80, 40)  # Rectangle around text
        if option_rect.collidepoint(x, y):
            print(option)
            if option == 'first':
                mode1 = True
                mode2 = mode3 = False
            elif option == 'second':
                mode2 = True
                mode1 = mode3 = False
            elif option == 'third':
                mode3 = True
                mode1 = mode2 = False
            return option
    return None


# with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh, \
     mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while RUNNING:
        if time.time() - start_time > run_time:
            print("Time's up! Exiting...")
            RUNNING = False
        pitft.update()
        for event in pygame.event.get():
            if event.type is MOUSEBUTTONDOWN:
                if in_start_menu:
                    x, y = pygame.mouse.get_pos()
                    selected_option = check_option_selection(x, y)
                    if selected_option is not None:
                        in_start_menu = False
                else:
                    # Handle other events during real-time detection if necessary
                    pass
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                RUNNING = False

        if in_start_menu:
            display_start_menu()
        else:
            # time.sleep(0.05)
            frame = picam2.capture_array()
            if frame is None:
                print("Unable to capture frame, please check the camera.")
                break

            if frame_count%frame_interval == 0:
                # Convert frame to RGB and process with MediaPipe
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                # results = holistic.process(image)
                face_results = face_mesh.process(image)
                hand_results = hands.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                frame_count = 0
            else:
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # Comment out the following line to see if no frames are extracted. Currently, two frames are extracted.
            # frame_count += 1

            # Draw landmarks for face, hands, and pose
            # mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
            #                             mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
            #                             mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1))
            # mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            #                             mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
            #                             mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2))
            # mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            #                             mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
            #                             mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2))
            # mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
            #                             mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
            #                             mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))


            if face_results.multi_face_landmarks:
                for face_landmarks in face_results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION,
                        mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                        mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1))

            # Draw hand landmarks
            display_hand = False
            gesture = "Unknown Gesture"
            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                        mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2))
                    # 检测手势
                    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
                    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
                    index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]

                    if index_tip.y < index_mcp.y and middle_tip.y > index_mcp.y and ring_tip.y > index_mcp.y:
                        gesture = "Gesture 1"
                    elif index_tip.y < index_mcp.y and middle_tip.y < index_mcp.y and ring_tip.y > index_mcp.y:
                        gesture = "Gesture 2"
                    elif index_tip.y < index_mcp.y and middle_tip.y < index_mcp.y and ring_tip.y < index_mcp.y:
                        gesture = "Gesture 3"
                    else:
                        gesture = "Unknown Gesture"

                    # print(f"Detected Gesture: {gesture}")
                    display_hand = True
                    # cv2.putText(image, gesture, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    # Reset countdown if gesture doesn't match new_mode
                    if gesture != "Unknown Gesture" and \
                    (new_mode == "Mode 1" and gesture != "Gesture 1" or \
                    new_mode == "Mode 2" and gesture != "Gesture 2" or \
                    new_mode == "Mode 3" and gesture != "Gesture 3"):
                        print("Reset countdown")
                        countdown_active = False
                        new_mode = None

                    # Check and switch modes based on detected gesture
                    if gesture == "Gesture 1" and not mode1:
                        new_mode = "Mode 1"
                    elif gesture == "Gesture 2" and not mode2:
                        new_mode = "Mode 2"
                    elif gesture == "Gesture 3" and not mode3:
                        new_mode = "Mode 3"

                    if new_mode and not countdown_active:
                        countdown_active = True
                        countdown_start_time = time.time()



            # Convert image to Pygame surface and render
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = np.rot90(image)
            surface = pygame.surfarray.make_surface(image)
            screen.blit(surface, (0, 0))

            # Render gesture type to screen
            if display_hand:
                gesture_text = font.render(f"{gesture}", True, RED)  # Red text
                gesture_rect = gesture_text.get_rect(topleft=(0, 0))
                screen.blit(gesture_text, gesture_rect) # Position at the top left

            # Render FPS to screen
            fps_text = font.render(f"FPS: {fps:.2f}", True, RED)  # Red text
            fps_rect = fps_text.get_rect(topright=(320, 0))
            screen.blit(fps_text, fps_rect) # Position at the top right

            # Inside the main loop, after rendering the gesture and FPS
            if mode1:
                mode_text = font.render("Mode 1", True, RED)
                mode_rect = mode_text.get_rect(bottomleft=(0, 240))
                screen.blit(mode_text, mode_rect)
            elif mode2:
                mode_text = font.render("Mode 2", True, RED)
                mode_rect = mode_text.get_rect(bottomleft=(0, 240))
                screen.blit(mode_text, mode_rect)
            elif mode3:
                mode_text = font.render("Mode 3", True, RED)
                mode_rect = mode_text.get_rect(bottomleft=(0, 240))
                screen.blit(mode_text, mode_rect)

            # Handle countdown
            if countdown_active:
                current_time = time.time()
                elapsed_time = current_time - countdown_start_time
                remaining_time = countdown_duration - int(elapsed_time)

                # Display countdown
                if remaining_time > 0:
                    countdown_text = font_mid.render(f"Will change to {new_mode}", True, RED)
                    countdown_rect = countdown_text.get_rect(center=(160, 40))
                    screen.blit(countdown_text, countdown_rect)

                    number_text = font_big.render(f"{str(remaining_time)}......", True, RED)
                    number_rect = number_text.get_rect(topright=(320, 60))
                    screen.blit(number_text, number_rect)
                else:
                    # Apply the new mode after countdown
                    if new_mode == "Mode 1":
                        mode1 = True
                        mode2 = mode3 = False
                    elif new_mode == "Mode 2":
                        mode2 = True
                        mode1 = mode3 = False
                    elif new_mode == "Mode 3":
                        mode3 = True
                        mode1 = mode2 = False
                        
                    # reset new_mode and countdown_active
                    new_mode = None
                    countdown_active = False  # Reset countdown


            pygame.display.update()

            # Increment frame count
            fps_frame_count += 1

            # Calculate and print FPS every second
            elapsed_time = time.time() - fps_start_time
            if elapsed_time >= 1.0:
                fps = fps_frame_count / elapsed_time
                # print(f"FPS: {fps:.2f}")
                fps_frame_count = 0
                fps_start_time = time.time()

        clock.tick(30)

GPIO.cleanup()
pygame.quit()
sys.exit(0)
