import cv2
import numpy as np
import config

def color_map():
    img = np.zeros(shape=(config.LABLE_SIZE, config.LABLE_SIZE))
    for i in range(0, config.CLASS_NUMBER):
        l = int(i * config.LABLE_SIZE / config.CLASS_NUMBER)
        r = int((i + 1) * config.LABLE_SIZE / config.CLASS_NUMBER)
        img[:, l:r] = i
    img = (img / config.CLASS_NUMBER * 255).astype(np.uint8)
    img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    cv2.imwrite('color.jpg', img)


if __name__ == '__main__':
    color_map()