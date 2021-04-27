import os
import paddle
import numpy as np
import paddle.fluid as fluid
import config
from dataset import TrainDataset, ValidDataset
from model import Unet
# from callback import StepTest
import random


def loss(x, label):
    # 平方差损失函数
    # loss = x - label
    # loss = 0.5 * loss * loss
    # loss = paddle.mean(loss)
    # return loss
    # 交叉熵损失函数
    one_hot = paddle.nn.functional.one_hot(label, num_classes=config.CLASS_NUMBER)
    loss = paddle.nn.functional.binary_cross_entropy(x, one_hot)
    return loss
    # 带权交叉熵损失函数
    # weight = None
    # # weight = paddle.to_tensor(config.WEIGHT)
    # one_hot = paddle.nn.functional.one_hot(label, num_classes=config.CLASS_NUMBER)
    # loss = paddle.nn.functional.binary_cross_entropy(x, one_hot, weight=weight)
    # return loss
    # 只统计目标类的带权交叉熵损失函数
    # weight = paddle.to_tensor(config.WEIGHT)
    # one_hot = paddle.nn.functional.one_hot(label, num_classes=config.CLASS_NUMBER)
    # p = paddle.sum(x * one_hot, axis=-1)
    # weight_map = paddle.mm(one_hot, weight)
    # loss = -paddle.log(p + 1e-6) * weight_map
    # loss = paddle.mean(loss)
    # return loss
    # 只统计分错的目标类的带权的交叉熵损失函数
    # weight = paddle.to_tensor(config.WEIGHT)
    # one_hot = paddle.nn.functional.one_hot(label, num_classes=config.CLASS_NUMBER) # 将lable转换为onehot
    # p = paddle.sum(x * one_hot, axis=-1) # one_hot与x点成，将每个像素目标类的概率保留，其他类的概率过滤，并reduce
    # lab = paddle.argmax(x, axis=-1) # 获得每个像素的分类结果
    # mask = (lab != label) # 获得分类错误的点
    # weight_map = paddle.mm(one_hot, weight) # 根据每个像素的类别，计算其权重
    # loss = -paddle.log(p + 1e-6) * weight_map * mask # 计算分类错误的像素的带权交叉熵
    # loss = paddle.mean(loss) # 求平均
    # return loss


def train():
    net = Unet(config.CLASS_NUMBER, act='leaky_relu')
    model = paddle.Model(net)
    # model.load('output/unet1')
    callback0 = paddle.callbacks.LRScheduler(by_step=True, by_epoch=False)
    train_dataset = TrainDataset('./train')
    valid_dataset = ValidDataset('./valid')
    scheduler = paddle.optimizer.lr.LinearWarmup(learning_rate=0.001, warmup_steps=len(train_dataset) // 1 * 3, start_lr=0, end_lr=0.001, verbose=True)
    # scheduler = paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=0.5, T_max=200, verbose=True)
    # optimizer = paddle.optimizer.SGD(learning_rate=scheduler, parameters=model.parameters())
    optimizer = paddle.optimizer.Adam(learning_rate=scheduler, parameters=model.parameters())
    model.prepare(optimizer, loss)
    model.fit(train_dataset, valid_dataset, batch_size=1, epochs=3, callbacks=[callback0], eval_freq=1, log_freq=1, shuffle=True)
    model.save('output/unet1')


if __name__ == "__main__":
    paddle.seed(1) # 设置全局随机种子
    random.seed(1) # 增加复现性
    train()