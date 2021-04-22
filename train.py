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
    one_hot = paddle.nn.functional.one_hot(label, num_classes=config.CLASS_NUMBER)
    loss = paddle.nn.functional.binary_cross_entropy(x, one_hot)
    return loss


def train():
    net = Unet(config.CLASS_NUMBER)
    model = paddle.Model(net)
    # model.load('output/unet1')
    callback = paddle.callbacks.LRScheduler(by_step=True, by_epoch=False)
    train_dataset = TrainDataset('./train')
    valid_dataset = ValidDataset('./valid')
    scheduler = paddle.optimizer.lr.LinearWarmup(learning_rate=0.01, warmup_steps=100, start_lr=0, end_lr=0.01, verbose=True)
    # scheduler = paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=0.5, T_max=200, verbose=True)
    optimizer = paddle.optimizer.SGD(learning_rate=scheduler, parameters=model.parameters())
    model.prepare(optimizer, loss)
    model.fit(train_dataset, valid_dataset, batch_size=1, epochs=30, callbacks=callback, eval_freq=1, log_freq=1)
    # model.save('output/unet1')


if __name__ == "__main__":
    train()