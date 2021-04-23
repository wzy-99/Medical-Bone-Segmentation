import os
import cv2
import paddle
import numpy as np
import config
from dataset import TestDataset
from model import Unet


def predict():
    testdataset = TestDataset('test')
    net = Unet(config.CLASS_NUMBER)
    model = paddle.Model(net)
    model.load('output/unet1')
    model.prepare()
    outs = model.predict(testdataset)
    outs = outs[0]
    for index, out in enumerate(outs):
        path = testdataset.indexs[index]
        img = cv2.imread(path)
        
        res = np.reshape(out, (224, 224, config.CLASS_NUMBER))
        # res = res * 254
        # res = res.astype(np.uint8)

        lab = (np.argmax(res, axis=-1) / config.CLASS_NUMBER * 255).astype(np.uint8)
        color = cv2.applyColorMap(lab, cv2.COLORMAP_JET)
        cv2.imwrite('result/' + str(index) + 'result.jpg', color)

        # for ch in range(config.CLASS_NUMBER):
        #     r = res[:, :, ch]
        #     lab = config.ID2LABEL[ch]
        #     cv2.imwrite('result/' + str(index) + 'result' + lab + '.jpg', r)
        
        cv2.imwrite('result/' + str(index) + 'origin.jpg', img)


if __name__ == '__main__':
    predict()
