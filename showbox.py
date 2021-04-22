import cv2


class ShowBox:
    def __init__(self, window_name='', window_size=(640, 480)):
        self.inited = False
        self.window_name = window_name
        self.window_size = window_size

    def init(self):
        if len(self.window_name) != 0:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, self.window_size[0], self.window_size[1])
        else:
            self.window_name = 'show'
            cv2.namedWindow('show', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('show', self.window_size[0], self.window_size[1])
    
    def show(self, img):
        if self.inited is False:
            self.init()
        cv2.imshow(self.window_name, img)


if __name__ == '__main__':
    import numpy as np
    showbox = ShowBox('test')
    img = np.random.randint(0, 255, size=(480, 640), dtype=np.uint8)
    showbox.show(img)
    cv2.waitKey(0)