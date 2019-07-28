# -*- coding: UTF-8 -*-

import numpy as np

import torch

from torch.autograd import Variable

import captcha_setting

import my_dataset

from captcha_cnn_model import CNN

import one_hot_encoding

import torch.nn as nn


def _where(cond, x, y):
    """
    code from :

        https://discuss.pytorch.org/t/how-can-i-do-the-operation-the-same-as-np-where/1329/8

    """
    cond = cond.float()

    return (cond * x) + ((1 - cond) * y)


# i-FGSM attack code
def i_fgsm(vimage, image, labels, criterion, net, alpha, iteration, epsilon):

    vimage.requires_grad = True

    # i-FGSM attack code
    for j in range(iteration):

        h_adv = net(vimage)

        loss = -criterion(h_adv, labels)

        net.zero_grad()

        loss.backward()

        vimage.grad.sign_()

        vimage = vimage - alpha * vimage.grad

        vimage = _where(vimage > image + epsilon, image + epsilon, vimage)

        vimage = _where(vimage < image - epsilon, image - epsilon, vimage)

        vimage = torch.clamp(vimage, -1, 1)

        vimage = Variable(vimage.data, requires_grad=True)

    return vimage


def decode_to(labels):

    c0 = captcha_setting.ALL_CHAR_SET[np.argmax(labels[0, 0:captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]

    c1 = captcha_setting.ALL_CHAR_SET[np.argmax(
        labels[0, captcha_setting.ALL_CHAR_SET_LEN:2 * captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]

    c2 = captcha_setting.ALL_CHAR_SET[np.argmax(
        labels[0, 2 * captcha_setting.ALL_CHAR_SET_LEN:3 * captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]

    c3 = captcha_setting.ALL_CHAR_SET[np.argmax(
        labels[0, 3 * captcha_setting.ALL_CHAR_SET_LEN:4 * captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]

    labels = '%s%s%s%s' % (c0, c1, c2, c3)

    return labels


def main():

    cnn = CNN()

    cnn.eval()

    cnn.load_state_dict(torch.load('model1.pkl'))

    print("load cnn net.")

    test_dataloader = my_dataset.get_test_data_loader()

    correct = 0

    total = 0

    epsilon = float(input("please input epsilon = "))

    iteration = int(input("please input iteration = "))

    criterion = nn.MultiLabelSoftMarginLoss()

    alpha = 1

    for i, (image, label) in enumerate(test_dataloader):

        vimage = Variable(image)

        labels = Variable(label.float())

        perturbed_data = i_fgsm(vimage, image, labels, criterion, cnn, alpha, iteration, epsilon)

        perturbed_label = cnn(perturbed_data)

        perturbed_label = decode_to(perturbed_label)

        true_label = one_hot_encoding.decode(label.numpy()[0])

        total += label.size(0)

        if (perturbed_label == true_label):

            correct += 1

        if (total % 10 == 0):

            print('Test Accuracy of the model on the %d test images: %f %%' % (total, 100 * correct / total))

    print('Test Accuracy of the model on the %d test images: %f %%' % (total, 100 * correct / total))


if __name__ == '__main__':
    main()

