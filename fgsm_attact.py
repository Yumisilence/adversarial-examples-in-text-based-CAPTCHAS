# -*- coding: UTF-8 -*-

import numpy as np

import torch

from torch.autograd import Variable

from torchvision import transforms

import captcha_setting

import my_dataset

from captcha_cnn_model import CNN

import one_hot_encoding

import torch.nn as nn

import os


# FGSM attack code
def fgsm_attack(image, epsilon, data_grad):

    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()

    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon * sign_data_grad

    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)

    # Return the perturbed image
    return perturbed_image

unloader = transforms.ToPILImage()


def main():
    cnn = CNN()

    cnn.eval()

    cnn.load_state_dict(torch.load('model1.pkl'))

    print("load cnn net.")

    test_dataloader = my_dataset.get_test_data_loader()

    correct = 0

    total = 0

    epsilon = float(input("please input epsilon = "))

    criterion = nn.MultiLabelSoftMarginLoss()

    for i, (images, labels) in enumerate(test_dataloader):

        vimage = Variable(images)

        labels = Variable(labels.float())

        vimage.requires_grad = True

        predict_labels = cnn(vimage)

        loss = criterion(predict_labels, labels)

        cnn.zero_grad()

        loss.backward()

        data_grad = vimage.grad.data

        perturbed_data = fgsm_attack(vimage, epsilon, data_grad)

        perturbed_label = cnn(perturbed_data)

        c0 = captcha_setting.ALL_CHAR_SET[np.argmax(
            perturbed_label[0, 0:captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]

        c1 = captcha_setting.ALL_CHAR_SET[np.argmax(
            perturbed_label[0, captcha_setting.ALL_CHAR_SET_LEN:2 * captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]

        c2 = captcha_setting.ALL_CHAR_SET[np.argmax(
            perturbed_label[0, 2 * captcha_setting.ALL_CHAR_SET_LEN:3 * captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]

        c3 = captcha_setting.ALL_CHAR_SET[np.argmax(
            perturbed_label[0, 3 * captcha_setting.ALL_CHAR_SET_LEN:4 * captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]

        perturbed_label = '%s%s%s%s' % (c0, c1, c2, c3)

        true_label = one_hot_encoding.decode(labels.numpy()[0])

        total += labels.size(0)

        if (perturbed_label == true_label):
            correct += 1

        if (total % 100 == 0):
            print('Test Accuracy of the model on the %d test images: %f %%' % (total, 100 * correct / total))

    print('Test Accuracy of the model on the %d test images: %f %%' % (total, 100 * correct / total))


if __name__ == '__main__':
    main()

