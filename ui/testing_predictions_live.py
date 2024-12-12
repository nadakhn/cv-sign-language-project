label_map = {'tur_N': 0, 'tur_O': 1, 'tur_P': 2, 'tur_R': 3, 'tur_S': 4, 'tur_T': 5, 'tur_U': 6, 'tur_V': 7, 'tur_Y': 8, 'tur_Z': 9, 'tur_D': 10, 'tur_E': 11, 'tur_F': 12, 'tur_G': 13, 'tur_H': 14, 'tur_I': 15, 'tur_J': 16, 'tur_K': 17, 'tur_L': 18, 'tur_M': 19, 'bis_Q': 20, 'bis_O': 21, 'bis_T': 22, 'bis_tur_C': 23, 'bis_D': 24, 'bis_U': 25, 'bis_M': 26, 'bis_K': 27, 'bis_B': 28, 'bis_Y': 29, 'bis_S': 30, 'bis_L': 31, 'bis_F': 32, 'bis_Z': 33, 'bis_E': 34, 'bis_G': 35, 'bis_P': 36, 'bis_A': 37, 'bis_X': 38, 'bis_V': 39, 'bis_R': 40, 'bis_W': 41, 'bis_N': 42, 'bis_I': 43, 'bis_H': 44, 'tur_A': 45, 'tur_B': 46}

import cv2
import numpy as np
import math
from cvzone.HandTrackingModule import HandDetector
import os

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Camera not accessible!")
    exit()

detector = HandDetector(maxHands=2)

offset = 45
img_size = 224

from tensorflow.keras.models import load_model
model = load_model('slt_model_rev2.h5')

# optional
save_processed = True
processed_folder = "processed_images"
if save_processed:
    os.makedirs(processed_folder, exist_ok=True)

expected_label = ""
total_signs = 0
correct_signs = 0
wrong_signs = 0

print("Press 'n' to enter a new expected label for validation.")

while True:
    success, img = cap.read()
    if not success:
        break

    # Detect hands in the frame
    hands, img_with_detections = detector.findHands(img, draw=False)

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

        img_landmarks_normalized = img_landmarks_pad / 255.0
        img_batch = np.expand_dims(img_landmarks_normalized, axis=0)  # shape: (1, 224, 224, 3)

        predictions = model.predict(img_batch)
        predicted_index = np.argmax(predictions)
        predicted_label = [label for label, index in label_map.items() if index == predicted_index][0]
        confidence = predictions[0][predicted_index]

        cv2.putText(
            img,
            f"Predicted: {predicted_label} ({confidence:.2f})",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        if expected_label:
            if predicted_label == expected_label:
                validation_message = "Correct!"
                correct_signs += 1
            else:
                validation_message = f"Wrong! Expected: {expected_label}"
                wrong_signs += 1

            cv2.putText(
                img,
                validation_message,
                (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255) if validation_message.startswith("Wrong") else (0, 255, 0),
                2,
            )
        total_signs += 1

    cv2.imshow("Live Feed", img)

    key = cv2.waitKey(1)
    if key == ord('q'):  # quit
        break
    elif key == ord('n'):  # input new expected label to check if prediction is correct
        expected_label = input("Enter the new expected label (e.g., 'bis_A'): ")

cap.release()
cv2.destroyAllWindows()

print(f"Total Signs Detected: {total_signs}")
print(f"Correct Predictions: {correct_signs}")
print(f"Wrong Predictions: {wrong_signs}")