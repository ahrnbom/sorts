import cv2
import numpy as np

def get_colors(num_classes):
    # Generates num_classes many distinct bright colors 
    class_colors = []
    for i in range(0, num_classes):
        # This can probably be written in a more elegant manner
        hue = 255*i/(num_classes+2)
        col = np.empty((1,1,3)).astype("uint8")
        col[0][0][0] = hue
        col[0][0][1] = 128 + (i%9)*9 # Saturation
        col[0][0][2] = 200 + (i%2)*55 # Value (brightness, I think?)
        cvcol = cv2.cvtColor(col, cv2.COLOR_HSV2BGR)
        col = (int(cvcol[0][0][0]), int(cvcol[0][0][1]), int(cvcol[0][0][2]))
        class_colors.append(col)
    return class_colors
