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
from cartoonize import caart

# ----------------- flag set -----------------
RUNNING = True
in_start_menu = True

# ----------------- time set -----------------
start_time = time.time()
run_time = 1000
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
# Define colors for the grid
BLUE = (0, 116, 217)      # #0074D9
NAVY = (0, 31, 63)        # #001F3F
GOLD = (255, 195, 0)      # #FFC300
GRAY = (179, 179, 179)    # #B3B3B3
BEIGE = (253, 245, 230)   # #FDF5E6
FOREST = (34, 139, 34)    # #228B22

font = pygame.font.Font(None, 30)
font_mid = pygame.font.Font(None, 45)
font_big = pygame.font.Font(None, 60)
options = ['glass', 'hat', 'all', 'sketch', 'cartoon', 'skeleton']
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
mode4 = False
mode5 = False
mode6 = False

# Global variables for countdown
countdown_active = False
countdown_start_time = 0
countdown_duration = 3  # 3 seconds countdown
new_mode = None

# read glasses image
glasses_img = cv2.imread('./assets/glasses.png', cv2.IMREAD_UNCHANGED)  # Include alpha channel
if glasses_img is None:
    print("Error: Glasses image not found. Please check the path.")
    exit()

hat_img = cv2.imread('./assets/hat.png', cv2.IMREAD_UNCHANGED)  # Include alpha channel
if glasses_img is None:
    print("Error: Hat image not found. Please check the path.")
    exit()

cigarette_img = cv2.imread('./assets/cigarette.png', cv2.IMREAD_UNCHANGED)  # Include alpha channel
if cigarette_img is None:
    print("Error: Cigarette image not found. Please check the path.")
    exit()

# Add near the image loading section at the beginning of the file
button_img = cv2.imread('./assets/button.png', cv2.IMREAD_UNCHANGED)  # Read button image
if button_img is None:
    print("Error: Button image not found. Please check the path.")
    exit()

# Display start menu with project title and options
def display_start_menu():
    # Fill top half with orange and bottom half with yellow
    screen.fill(ORANGE, rect=(0, 0, 320, 100))
    screen.fill(YELLOW, rect=(0, 100, 320, 140))
    
    # Draw circles at (160, 100)
    pygame.draw.circle(screen, YELLOW, (160, 80), 80)  # Larger yellow circle
    pygame.draw.circle(screen, ORANGE, (160, 80), 70)  # Smaller orange circle

    # Display title in the middle of the top half
    title_surface_b = font_big.render('Magic', True, WHITE)
    title_rect_b = title_surface_b.get_rect(center=(170, 65))
    screen.blit(title_surface_b, title_rect_b)

    title_surface = font_big.render('Magic', True, BLACK)
    title_rect = title_surface.get_rect(center=(160, 75))
    screen.blit(title_surface, title_rect)

    # Calculate dimensions for each grid cell
    cell_width = 320 // 3
    cell_height = 140 // 2  # (240-100) / 2 = 70

    for i in range(6):
        row = i // 3
        col = i % 3
        x = col * cell_width
        y = 100 + row * cell_height
        # screen.fill(colors[i], rect=(x, y, cell_width, cell_height))
        
        # Calculate the size and position of the button image
        button_size = (int(cell_width * 0.9), int(cell_height * 0.9))
        button_x = x + (cell_width - button_size[0]) // 2
        button_y = y + (cell_height - button_size[1]) // 2
        
        # Resize the button image
        resized_button = cv2.resize(button_img, button_size)
        # rotate the image 90 degrees clockwise
        rotated_button = cv2.rotate(resized_button, cv2.ROTATE_90_CLOCKWISE)
        
        # Correctly handle the transparent channel
        if rotated_button.shape[2] == 4:  # Check if there is an alpha channel
            # Separate BGR and alpha channels
            bgr = rotated_button[:, :, :3]
            alpha = rotated_button[:, :, 3]

            # Convert to RGB
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            # Create surface with alpha
            button_surface = pygame.Surface(rgb.shape[:-1], pygame.SRCALPHA)

            # Convert surface to a compatible format
            button_surface = button_surface.convert_alpha()  # Use convert_alpha() for surfaces with alpha

            pygame.surfarray.pixels3d(button_surface)[:] = rgb
            pygame.surfarray.pixels_alpha(button_surface)[:] = alpha

        screen.blit(button_surface, (button_x, button_y))

    # Display options
    for i, option in enumerate(options):
        row = i // 3
        col = i % 3
        x_position = col * cell_width + cell_width// 2
        y_position = 100 + row * cell_height + cell_height// 2
        
        option_surface = font.render(f'{option}', True, BEIGE)
        option_rect = option_surface.get_rect(center=(x_position, y_position))
        screen.blit(option_surface, option_rect)

    pygame.display.update()

# Function to check which option is selected
def check_option_selection(x, y):
    global mode1, mode2, mode3, mode4, mode5, mode6
    # Calculate dimensions for each grid cell
    cell_width = 320 // 3
    cell_height = 140 // 2  # (240-100) / 2 = 70
    
    for i, option in enumerate(options):
        row = i // 3
        col = i % 3
        x_position = col * cell_width + cell_width// 2
        y_position = 100 + row * cell_height + cell_height// 2
        
        # Create a rectangle around the button (40 pixels padding)
        option_rect = pygame.Rect(x_position - 40, y_position - 15, 80, 40)
        
        if option_rect.collidepoint(x, y):
            print(option)
            # Reset all modes
            mode1 = mode2 = mode3 = mode4 = mode5 = mode6 = False
            # Set the selected mode
            if option == 'glass':
                mode1 = True
            elif option == 'hat':
                mode2 = True
            elif option == 'all':
                mode3 = True
            elif option == 'sketch':
                mode4 = True
            elif option == 'cartoon':
                mode5 = True
            elif option == 'skeleton':
                mode6 = True
            return option
    return None


def sketch_image(img):
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply median blur
    gray = cv2.medianBlur(gray, 5)
    # Use Canny edge detection
    edges = cv2.Canny(gray, 50, 150)  # Adjust thresholds to enhance edges
    # Use Laplacian operator to enhance edges
    laplacian = cv2.Laplacian(gray, cv2.CV_8U, ksize=5)
    edges = cv2.bitwise_or(edges, laplacian)

    # Downsample the image
    color = cv2.bilateralFilter(img, 9, 250, 250)
    # Combine edges with color image
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    cartoon = cv2.bitwise_and(color, edges_colored)
    return edges_colored

# overlay image
def overlay_image(bg_image, overlay_image, x, y, overlay_size):
    overlay = cv2.resize(overlay_image, overlay_size)
    h, w, _ = overlay.shape
    alpha_overlay = overlay[:, :, 3] / 255.0
    alpha_bg = 1.0 - alpha_overlay

    # Ensure overlay position is within background image bounds
    if y < 0 or x < 0 or y + h > bg_image.shape[0] or x + w > bg_image.shape[1]:
        print("Overlay position is out of bounds, skipping overlay.")
        return

    for c in range(0, 3):
        bg_image[y:y+h, x:x+w, c] = (alpha_overlay * overlay[:, :, c] +
                                     alpha_bg * bg_image[y:y+h, x:x+w, c])


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

            # Draw hand landmarks
            display_hand = False
            gesture = "Unknown Gesture"
            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    # mp_drawing.draw_landmarks(
                    #     image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    #     mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                    #     mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2))
                    # Detect gestures
                    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
                    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
                    index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]

                    # Gesture 1: Only index finger up
                    if (index_tip.y < index_mcp.y and 
                        middle_tip.y > index_mcp.y and 
                        ring_tip.y > index_mcp.y and 
                        pinky_tip.y > index_mcp.y and 
                        thumb_tip.x > index_mcp.x):
                        gesture = "Gesture 1"
                    # Gesture 2: Index and middle fingers up
                    elif (index_tip.y < index_mcp.y and 
                          middle_tip.y < index_mcp.y and 
                          ring_tip.y > index_mcp.y and 
                          pinky_tip.y > index_mcp.y and 
                          thumb_tip.x > index_mcp.x):
                        gesture = "Gesture 2"
                    # Gesture 3: Index, middle, and ring fingers up
                    elif (index_tip.y < index_mcp.y and 
                          middle_tip.y < index_mcp.y and 
                          ring_tip.y < index_mcp.y and 
                          pinky_tip.y > index_mcp.y and 
                          thumb_tip.x > index_mcp.x):
                        gesture = "Gesture 3"
                    # Gesture 4: All fingers except thumb up
                    elif (index_tip.y < index_mcp.y and 
                          middle_tip.y < index_mcp.y and 
                          ring_tip.y < index_mcp.y and 
                          pinky_tip.y < index_mcp.y and 
                          thumb_tip.x > index_mcp.x):
                        gesture = "Gesture 4"
                    # Gesture 5: All fingers up
                    elif (thumb_tip.x < index_mcp.x and 
                          index_tip.y < index_mcp.y and 
                          middle_tip.y < index_mcp.y and 
                          ring_tip.y < index_mcp.y and 
                          pinky_tip.y < index_mcp.y):
                        gesture = "Gesture 5"
                    # Gesture 6: Only thumb and pinky up (phone gesture)
                    elif (thumb_tip.x < index_mcp.x and 
                          pinky_tip.y < index_mcp.y and
                          index_tip.y > index_mcp.y and 
                          middle_tip.y > index_mcp.y and 
                          ring_tip.y > index_mcp.y):
                        gesture = "Gesture 6"
                    else:
                        gesture = "Unknown Gesture"

                    # print(f"Detected Gesture: {gesture}")
                    display_hand = True
                    # cv2.putText(image, gesture, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    # Reset countdown if gesture doesn't match new_mode
                    if gesture != "Unknown Gesture" and \
                    (new_mode == "Mode 1" and gesture != "Gesture 1" or \
                    new_mode == "Mode 2" and gesture != "Gesture 2" or \
                    new_mode == "Mode 3" and gesture != "Gesture 3" or \
                    new_mode == "Mode 4" and gesture != "Gesture 4" or \
                    new_mode == "Mode 5" and gesture != "Gesture 5" or \
                    new_mode == "Mode 6" and gesture != "Gesture 6"):
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
                    elif gesture == "Gesture 4" and not mode4:
                        new_mode = "Mode 4"
                    elif gesture == "Gesture 5" and not mode5:
                        new_mode = "Mode 5"
                    elif gesture == "Gesture 6" and not mode6:
                        new_mode = "Mode 6"

                    if new_mode and not countdown_active:
                        countdown_active = True
                        countdown_start_time = time.time()


            # Inside the main loop, after rendering the gesture and FPS
            if mode1:
                if face_results.multi_face_landmarks:
                    for face_landmarks in face_results.multi_face_landmarks:
                        h, w, _ = image.shape
                        left_eye = face_landmarks.landmark[33]  # Left eye keypoint
                        right_eye = face_landmarks.landmark[263]  # Right eye keypoint
                        x1 = int(left_eye.x * w)
                        y1 = int(left_eye.y * h)
                        x2 = int(right_eye.x * w)
                        y2 = int(right_eye.y * h)
                        glasses_width = x2 - x1
                        glasses_height = int(glasses_width * glasses_img.shape[0] / glasses_img.shape[1])
                        # Ensure glasses position is within image bounds
                        if y1 - int(glasses_height / 2) >= 0 and x1 >= 0 and (y1 + glasses_height) <= h and (x1 + glasses_width) <= w:
                            overlay_image(image, glasses_img, x1, y1 - int(glasses_height / 2), (glasses_width, glasses_height))
                        else:
                            print("Glasses position is out of bounds, skipping overlay.")
            elif mode2:
                if face_results.multi_face_landmarks:
                    for face_landmarks in face_results.multi_face_landmarks:
                        h, w, _ = image.shape
                        # Use forehead keypoint from face mesh
                        forehead = face_landmarks.landmark[10]  # Top of the head keypoint
                        chin = face_landmarks.landmark[152]  # Chin keypoint
                        left_cheek = face_landmarks.landmark[234]  # Left cheek keypoint
                        right_cheek = face_landmarks.landmark[454]  # Right cheek keypoint

                        # Calculate face width and height
                        face_width = int(abs(right_cheek.x - left_cheek.x) * w)
                        face_height = int(abs(chin.y - forehead.y) * h)

                        # Calculate head tilt angle and invert
                        delta_x = right_cheek.x - left_cheek.x
                        delta_y = right_cheek.y - left_cheek.y
                        angle = -np.arctan2(delta_y, delta_x) * (180.0 / np.pi)

                        # Determine hat position and size
                        scale_factor = 1.5  # Adjust this factor to change hat size
                        x_hat = int(forehead.x * w) - int(face_width * scale_factor) // 2
                        y_hat = int(forehead.y * h) - int(face_height * scale_factor) // 2 + int(0.2 * face_height)  # Move hat down

                        hat_width = int(face_width * scale_factor)
                        hat_height = int(hat_width * hat_img.shape[0] / hat_img.shape[1])

                        if y_hat >= 0 and x_hat >= 0 and (y_hat + hat_height) <= h and (x_hat + hat_width) <= w:
                            # Calculate rotation center
                            center = (hat_img.shape[1] // 2, hat_img.shape[0] // 2)
                            # Calculate rotation matrix
                            rotated_hat = cv2.getRotationMatrix2D(center, angle, 1.0)
                            # Calculate rotated bounding box
                            cos = np.abs(rotated_hat[0, 0])
                            sin = np.abs(rotated_hat[0, 1])
                            new_w = int((hat_img.shape[0] * sin) + (hat_img.shape[1] * cos))
                            new_h = int((hat_img.shape[0] * cos) + (hat_img.shape[1] * sin))
                            # Adjust translation part of rotation matrix
                            rotated_hat[0, 2] += (new_w / 2) - center[0]
                            rotated_hat[1, 2] += (new_h / 2) - center[1]
                            # Rotate image
                            rotated_hat_img = cv2.warpAffine(hat_img, rotated_hat, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
                            overlay_image(image, rotated_hat_img, x_hat, y_hat, (hat_width, hat_height))
                        else:
                            print("Hat position is out of bounds, skipping overlay.")
            elif mode3:
                if face_results.multi_face_landmarks:
                    for face_landmarks in face_results.multi_face_landmarks:
                        h, w, _ = image.shape

                        left_eye = face_landmarks.landmark[33]  # Left eye keypoint
                        right_eye = face_landmarks.landmark[263]  # Right eye keypoint
                        x1 = int(left_eye.x * w)
                        y1 = int(left_eye.y * h)
                        x2 = int(right_eye.x * w)
                        y2 = int(right_eye.y * h)
                        glasses_width = x2 - x1
                        glasses_height = int(glasses_width * glasses_img.shape[0] / glasses_img.shape[1])
                        # Ensure glasses position is within image bounds
                        if y1 - int(glasses_height / 2) >= 0 and x1 >= 0 and (y1 + glasses_height) <= h and (x1 + glasses_width) <= w:
                            overlay_image(image, glasses_img, x1, y1 - int(glasses_height / 2), (glasses_width, glasses_height))
                        else:
                            # print("Glasses position is out of bounds, skipping overlay.")
                            pass
                        
                        forehead = face_landmarks.landmark[10]  # Top of the head keypoint
                        chin = face_landmarks.landmark[152]  # Chin keypoint
                        left_cheek = face_landmarks.landmark[234]  # Left cheek keypoint
                        right_cheek = face_landmarks.landmark[454]  # Right cheek keypoint
                        # Calculate face width and height
                        face_width = int(abs(right_cheek.x - left_cheek.x) * w)
                        face_height = int(abs(chin.y - forehead.y) * h)

                        # Calculate head tilt angle and invert
                        delta_x = right_cheek.x - left_cheek.x
                        delta_y = right_cheek.y - left_cheek.y
                        angle = -np.arctan2(delta_y, delta_x) * (180.0 / np.pi)

                        # Determine hat position and size
                        scale_factor = 1.5  # Adjust this factor to change hat size
                        x_hat = int(forehead.x * w) - int(face_width * scale_factor) // 2
                        y_hat = int(forehead.y * h) - int(face_height * scale_factor) // 2 + int(0.2 * face_height)  # Move hat down

                        hat_width = int(face_width * scale_factor)
                        hat_height = int(hat_width * hat_img.shape[0] / hat_img.shape[1])

                        if y_hat >= 0 and x_hat >= 0 and (y_hat + hat_height) <= h and (x_hat + hat_width) <= w:
                            # Calculate rotation center
                            center = (hat_img.shape[1] // 2, hat_img.shape[0] // 2)
                            # Calculate rotation matrix
                            rotated_hat = cv2.getRotationMatrix2D(center, angle, 1.0)
                            # Calculate rotated bounding box
                            cos = np.abs(rotated_hat[0, 0])
                            sin = np.abs(rotated_hat[0, 1])
                            new_w = int((hat_img.shape[0] * sin) + (hat_img.shape[1] * cos))
                            new_h = int((hat_img.shape[0] * cos) + (hat_img.shape[1] * sin))
                            # Adjust translation part of rotation matrix
                            rotated_hat[0, 2] += (new_w / 2) - center[0]
                            rotated_hat[1, 2] += (new_h / 2) - center[1]
                            # Rotate image
                            rotated_hat_img = cv2.warpAffine(hat_img, rotated_hat, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
                            overlay_image(image, rotated_hat_img, x_hat, y_hat, (hat_width, hat_height))
                        else:
                            # print("Hat position is out of bounds, skipping overlay.")
                            pass

                        # Use mouth keypoints from face mesh
                        mouth_left = face_landmarks.landmark[61]  # Left corner of the mouth
                        mouth_right = face_landmarks.landmark[291]  # Right corner of the mouth
                        mouth_center = face_landmarks.landmark[13]  # Center of the mouth

                        # Calculate mouth width and height with a scaling factor
                        scale_factor = 2.5  # Increase this factor to make the image larger
                        mouth_width = int(abs(mouth_right.x - mouth_left.x) * w * scale_factor)
                        mouth_height = int(mouth_width * cigarette_img.shape[0] / cigarette_img.shape[1])

                        # Determine cigarette position
                        x_cigarette = int(mouth_center.x * w) - mouth_width
                        y_cigarette = int(mouth_center.y * h) - mouth_height // 2 - 10

                        # Horizontal flip cigarette image to correct orientation
                        flipped_cigarette = cv2.flip(cigarette_img, 1)

                        # Ensure cigarette position is within image bounds
                        if y_cigarette >= 0 and x_cigarette >= 0 and (y_cigarette + mouth_height) <= h and (x_cigarette + mouth_width) <= w:
                            overlay_image(image, flipped_cigarette, x_cigarette, y_cigarette, (mouth_width, mouth_height))
                        else:
                            # print("Cigarette position is out of bounds, skipping overlay.")
                            pass
            elif mode4:
                # Apply sketch effect
                cartoon_frame = sketch_image(image)
                image = cartoon_frame
            elif mode5:
                # Apply the cartoon effect using the caart function
                cartoon_frame = caart(image)
                # Display the cartoonized image
                image = cartoon_frame
            elif mode6:
                if face_results.multi_face_landmarks:
                    for face_landmarks in face_results.multi_face_landmarks:
                        mp_drawing.draw_landmarks(
                            image, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION,
                            mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                            mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1))
                if hand_results.multi_hand_landmarks:
                    for hand_landmarks in hand_results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                            mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2))

            # Convert image to Pygame surface and render
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = np.rot90(image)
            surface = pygame.surfarray.make_surface(image)
            screen.blit(surface, (0, 0))
            
            # must behind the surface rendering 
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
            elif mode4:
                mode_text = font.render("Mode 4", True, RED)
                mode_rect = mode_text.get_rect(bottomleft=(0, 240))
                screen.blit(mode_text, mode_rect)
            elif mode5:
                mode_text = font.render("Mode 5", True, RED)
                mode_rect = mode_text.get_rect(bottomleft=(0, 240))
                screen.blit(mode_text, mode_rect)
            elif mode6:
                mode_text = font.render("Mode 6", True, RED)
                mode_rect = mode_text.get_rect(bottomleft=(0, 240))
                screen.blit(mode_text, mode_rect)

            # Render gesture type to screen
            if display_hand:
                gesture_text = font.render(f"{gesture}", True, RED)  # Red text
                gesture_rect = gesture_text.get_rect(topleft=(0, 0))
                screen.blit(gesture_text, gesture_rect) # Position at the top left

            # Render FPS to screen
            fps_text = font.render(f"FPS: {fps:.2f}", True, RED)  # Red text
            fps_rect = fps_text.get_rect(topright=(320, 0))
            screen.blit(fps_text, fps_rect) # Position at the top right

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
                        mode2 = mode3 = mode4 = mode5 = mode6 = False
                    elif new_mode == "Mode 2":
                        mode2 = True
                        mode1 = mode3 = mode4 = mode5 = mode6 = False
                    elif new_mode == "Mode 3":
                        mode3 = True
                        mode1 = mode2 = mode4 = mode5 = mode6 = False
                    elif new_mode == "Mode 4":
                        mode4 = True
                        mode1 = mode2 = mode3 = mode5 = mode6 = False
                    elif new_mode == "Mode 5":
                        mode5 = True
                        mode1 = mode2 = mode3 = mode4 = mode6 = False
                    elif new_mode == "Mode 6":
                        mode6 = True
                        mode1 = mode2 = mode3 = mode4 = mode5 = False
                        
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
