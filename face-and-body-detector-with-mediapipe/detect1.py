from picamera2 import Picamera2
import cv2
import mediapipe as mp
import pygame
import numpy as np
import os
import sys
import RPi.GPIO as GPIO
import time
from pygame.locals import *

def initialize_pygame():
    # Set SDL environment to ensure PiTFT display
    os.putenv('SDL_VIDEODRIVER', 'fbcon')  # Frame cache
    os.putenv('SDL_FBDEV', '/dev/fb0')
    os.putenv('SDL_MOUSEDRV', 'dummy')
    os.putenv('SDL_MOUSEDEV', '/dev/null')
    os.putenv('DISPLAY', '')

    # Initialize pygame
    pygame.init()
    screen = pygame.display.set_mode((320, 240))  # Adjust according to PiTFT resolution
    pygame.mouse.set_visible(False)
    return screen

def display_start_menu(screen, font):
    # Display start menu with project title and options
    screen.fill((0, 0, 0))  # Black background
    title_surface = font.render('Magic', True, (255, 255, 255))
    screen.blit(title_surface, (120, 60))

    # Define button text and positions
    options = ['1', '2', '3']
    positions = [(140, 120), (140, 150), (140, 180)]
    for i, (option, pos) in enumerate(zip(options, positions)):
        option_surface = font.render(option, True, (255, 255, 255))
        screen.blit(option_surface, pos)
    pygame.display.update()

    return options, positions

def check_option_selection(options, positions, x, y):
    # Detect which button is pressed
    for option, pos in zip(options, positions):
        option_rect = pygame.Rect(pos[0] - 20, pos[1] - 10, 40, 20)  # Rectangle around text
        print(1)
        if option_rect.collidepoint(x, y):
            print(2)
            return option
    return None

def main():
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(27, GPIO.IN, pull_up_down=GPIO.PUD_UP)

    def GPIO27_callback(channel):
        global RUNNING
        print("Quit....")
        RUNNING = False

    GPIO.add_event_detect(27, GPIO.FALLING, callback=GPIO27_callback, bouncetime=200)
    global RUNNING
    RUNNING = True
    pitft = pygame.display
    screen = initialize_pygame()
    font = pygame.font.Font(None, 30)
    options, positions = display_start_menu(screen, font)
    in_start_menu = True

    # Initialize PiCamera2 and MediaPipe
    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(main={"size": (320, 240)}))
    picam2.start()

    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic
    clock = pygame.time.Clock()
    start_time = time.time()
    run_time = 100

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while RUNNING:
            for event in pygame.event.get():
                if event.type == MOUSEBUTTONDOWN:
                    x, y = pygame.mouse.get_pos()
                    if in_start_menu:
                        selected_option = check_option_selection(options, positions, x, y)
                        if selected_option is not None:
                            in_start_menu = False
                    else:
                        # Handle other events during real-time detection if necessary
                        pass
                if event.type == pygame.QUIT:
                    RUNNING = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                    RUNNING = False

            if in_start_menu:
                display_start_menu(screen, font)
            else:
                if time.time() - start_time > run_time:
                    print("Time's up! Exiting...")
                    RUNNING = False

                frame = picam2.capture_array()
                if frame is None:
                    print("Unable to capture frame, please check the camera.")
                    break

                # Convert frame to RGB and process with MediaPipe
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = holistic.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # Draw landmarks for face, hands, and pose
                mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                                          mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                                          mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1))
                mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                                          mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2))
                mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                          mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2))
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                                          mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

                # Convert image to Pygame surface and render
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = np.rot90(image)
                surface = pygame.surfarray.make_surface(image)
                screen.blit(surface, (0, 0))
                pygame.display.update()

            clock.tick(30)

if __name__ == "__main__":
    main()
    GPIO.cleanup()
    pygame.quit()
    sys.exit()
