import os
import paddle
import numpy as np
import paddle.fluid as fluid
import config
from dataset import TrainDataset, ValidDataset
from model import Unet


def loss(x, label):
    # loss = x - label
    # loss = 0.5 * loss * loss
    # loss = paddle.mean(loss)
    # return loss
    loss = paddle.nn.functional.binary_cross_entropy(x, label)
    return loss


def train():
    net = Unet(config.CLASS_NUMBER)
    model = paddle.Model(net)
    # model.load('output/unet1')
    callback = paddle.callbacks.LRScheduler(by_step=True, by_epoch=False)
    train_dataset = TrainDataset('./train', class_number=config.CLASS_NUMBER)
    valid_dataset = ValidDataset('./valid', class_number=config.CLASS_NUMBER)
    scheduler = paddle.optimizer.lr.LinearWarmup(learning_rate=0.01, warmup_steps=100, start_lr=0, end_lr=0.01)
    # scheduler = paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=0.5, T_max=200, verbose=True)
    optimizer = paddle.optimizer.SGD(learning_rate=scheduler, parameters=model.parameters())
    model.prepare(optimizer, loss)
    model.fit(train_dataset, valid_dataset, batch_size=1, epochs=1, callbacks=callback, eval_freq=1)
    # model.save('output/unet1')


if __name__ == "__main__":
    train()