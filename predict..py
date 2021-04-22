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
        
        res = np.reshape(out, (config.CLASS_NUMBER, 224, 224))
        res = res * 254
        res = res.astype(np.uint8)

        for ch, r in enumerate(res):
            lab = config.ID2LABEL[ch]
            cv2.imwrite('result/' + str(index) + 'result' + lab + '.jpg', r)
        
        cv2.imwrite('result/' + str(index) + 'origin.jpg', img)


if __name__ == '__main__':
    predict()
