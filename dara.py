import cv2
import numpy as np
import webcolors
from webcolors import rgb_to_name

class Video(object):
    def __init__(self):
        self.video=cv2.VideoCapture(0)
        
    def __del__(self):
        self.video.release()
        
    def get_frame(self):
        ret,frame=self.video.read()

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply a mosaic effect
        mosaic = cv2.resize(gray, None, fx=0.1, fy=0.1, interpolation=cv2.INTER_NEAREST)
        mosaic = cv2.resize(
            mosaic, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST
        )

        # Threshold to create a binary mask of mosaic areas
        _, binary_mask = cv2.threshold(mosaic, 150, 255, cv2.THRESH_BINARY)

        # Invert the binary mask
        inverted_mask = cv2.bitwise_not(binary_mask)

        # Use the inverted mask to mask out the mosaic areas in the original frame
        result = cv2.bitwise_and(frame, frame, mask=inverted_mask)

        # Calculate the percentage of completion
        total_mosaic_pixels = np.sum(binary_mask > 0)
        if total_mosaic_pixels != 0:
            colored_pixels = np.sum(inverted_mask > 0)
            completion_percentage = (colored_pixels / total_mosaic_pixels) * 100
        else:
            completion_percentage = 100  # No mosaic pixels, assume it's fully colored

        # Get the color being used
        color_index = int(
            completion_percentage / 33.33
        )  # 3 colors, so each covers roughly 33.33%
        colored_pixels = result[inverted_mask > 0]
        avg_color = np.mean(colored_pixels, axis=0)

        # Get the color name
        color_name = self.get_color_name(avg_color.astype(int))

        # Draw progress information on the frame
        progress_info = f"Progress: {completion_percentage:.2f}%"
        cv2.putText(
            result, progress_info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
        )
        cv2.putText(
            result,
            f"Color: {color_name}",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )
        ret,jpg=cv2.imencode('.jpg',result)
        return jpg.tobytes()

    def get_color_name(self, rgb):
        closest_color = min(
                webcolors.CSS3_HEX_TO_NAMES,
                key=lambda name: sum(
                    (a - b) ** 2 for a, b in zip(webcolors.hex_to_rgb(name), rgb)
                ),
            )
        return webcolors.CSS3_HEX_TO_NAMES[closest_color]
