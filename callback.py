import paddle
import cv2
import numpy as np
from dataset import TestDataset
import config

class StepTest(paddle.callbacks.Callback):
    def __init__(self):
        self.testdataset = TestDataset('test')

    def on_train_batch_end(self, step, logs=None):
        outs = self.model.predict(self.testdataset)
        outs = outs[0]
        for index, out in enumerate(outs):
            path = self.testdataset.indexs[index]
            img = cv2.imread(path)
            
            # # 输出热力图
            # res = np.reshape(out, (config.LABLE_SIZE, config.LABLE_SIZE, config.CLASS_NUMBER))
            # lab = (np.argmax(res, axis=-1) / config.CLASS_NUMBER * 255).astype(np.uint8)
            # color = cv2.applyColorMap(lab, cv2.COLORMAP_JET)
            # cv2.imwrite('result/' + str(index) + 'result.jpg', color)

            # 输出每张图
            res = np.reshape(out, (config.LABLE_SIZE, config.LABLE_SIZE, config.CLASS_NUMBER))
            res = res * 254
            res = res.astype(np.uint8)
            # for ch in range(config.CLASS_NUMBER):
            #     r = res[:, :, ch]
            #     lab = config.ID2LABEL[ch]
            #     cv2.imwrite('step_log/' + str(step) + 'result' + lab + '.jpg', r)
            ch = 1
            r = res[:, :, ch]
            lab = config.ID2LABEL[ch]
            cv2.imwrite('step_log/' + str(step) + 'result' + lab + '.jpg', r)
