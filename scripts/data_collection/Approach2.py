import cv2
import numpy as np
import math
from cvzone.HandTrackingModule import HandDetector
import os

# Initialize video capture and hand detector
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)

offset = 45
img_size = 224

name = "bis_A"

# Create folders to save the processed images
folders = {
    "just_hands": "data/just_hands",
    "with_hands": "data/with_hands",
    "black_landmarks": "data/black_landmarks"
}

for folder in folders.values():
    os.makedirs(folder, exist_ok=True)

# Initialize Camera and Hand Detector
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)
counter = 0

while True:
    success, img = cap.read()
    if not success:
        break

    # Detect hands in the frame
    hands, img = detector.findHands(img, draw=False)

    if hands:
        # Initialize variables for the combined bounding box
        x_min, y_min = float('inf'), float('inf')
        x_max, y_max = float('-inf'), float('-inf')

        # Loop through detected hands to compute the combined bounding box
        for hand in hands:
            x, y, w, h = hand['bbox']
            x_min = min(x_min, x - offset)
            y_min = min(y_min, y - offset)
            x_max = max(x_max, x + w + offset)
            y_max = max(y_max, y + h + offset)

        # Ensure the bounding box is within image boundaries
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(img.shape[1], x_max)
        y_max = min(img.shape[0], y_max)

        # Draw the single combined bounding box
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 0, 255), 2)

        img_just_hands = img[int(y_min):int(y_max), int(x_min):int(x_max)].copy()

        # Black background for landmarks only
        img_landmarks = np.zeros_like(img)
        # Draw landmarks for both hands
        for hand in hands:
            lm_list = hand['lmList']  # List of hand landmarks
            for lm in lm_list:
                cx, cy = lm[:2]
                cv2.circle(img, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
                cv2.circle(img_landmarks, (cx, cy), 5, (0, 255, 0), cv2.FILLED)

        # Crop the image based on the combined bounding box
        img_crop = img[int(y_min):int(y_max), int(x_min):int(x_max)]
        img_landmarks_crop = img_landmarks[int(y_min):int(y_max), int(x_min):int(x_max)]

        # Resize and pad the cropped image to maintain aspect ratio
        aspect_ratio = (y_max - y_min) / (x_max - x_min)
        img_pad = np.zeros((img_size, img_size, 3), np.uint8)
        img_landmarks_pad = np.zeros((img_size, img_size, 3), np.uint8)

        if aspect_ratio > 1:
            # Height is greater; resize width
            k = img_size / (y_max - y_min)
            new_w = math.ceil(k * (x_max - x_min))
            img_resize = cv2.resize(img_crop, (new_w, img_size))
            img_landmarks_resize = cv2.resize(img_landmarks_crop, (new_w, img_size))
            w_offset = (img_size - new_w) // 2
            img_pad[:, w_offset:w_offset + new_w] = img_resize
            img_landmarks_pad[:, w_offset:w_offset + new_w] = img_landmarks_resize
        else:
            # Width is greater; resize height
            k = img_size / (x_max - x_min)
            new_h = math.ceil(k * (y_max - y_min))
            img_resize = cv2.resize(img_crop, (img_size, new_h))
            img_landmarks_resize = cv2.resize(img_landmarks_crop, (img_size, new_h))
            h_offset = (img_size - new_h) // 2
            img_pad[h_offset:h_offset + new_h, :] = img_resize
            img_landmarks_pad[h_offset:h_offset + new_h, :] = img_landmarks_resize

        # Display the processed images
        cv2.imshow("Cropped Image", img_crop)
        cv2.imshow("Processed Image", img_pad)
        cv2.imshow("Just hands", img_just_hands)
        cv2.imshow("Landmarks", img_landmarks_pad)

    # Show the original image with landmarks and combined bounding box
    cv2.imshow("Image", img)

    # Handle key press events
    key = cv2.waitKey(1)
    # Break the loop when 'q' is pressed
    if key == ord('q'):
        break
    # Save the processed image when 's' is pressed
    if key == ord('s'):
        counter += 1
        file_path_with_hands = f"{folders['with_hands']}/{name} ({counter}).jpg"
        file_path_just_hands = f"{folders["just_hands"]}/{name} ({counter}).jpg"
        file_path_black_landmarks = f"{folders['black_landmarks']}/{name} ({counter}).jpg"
        cv2.imwrite(file_path_with_hands, img_pad)
        cv2.imwrite(file_path_just_hands, img_just_hands)
        cv2.imwrite(file_path_black_landmarks, img_landmarks_pad)
        print(f"Image saved as: {file_path_with_hands}")
        print(f"Image saved as: {file_path_just_hands}")
        print(f"Image saved as: {file_path_black_landmarks}")

cap.release()
cv2.destroyAllWindows()
