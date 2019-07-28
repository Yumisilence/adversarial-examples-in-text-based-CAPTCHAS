# -*- coding: UTF-8 -*-

import numpy as np

import torch

from torch.autograd import Variable

import captcha_setting

import my_dataset

from captcha_cnn_model import CNN

import one_hot_encoding

import copy

from torch.autograd.gradcheck import zero_gradients


def deepfool(image, net, num_classes=10, overshoot=0.02, max_iter=50):

    """
       :param image: Image of size HxWx3
       :param net: network (input: images, output: values of activation **BEFORE** softmax).
       :param num_classes: num_classes (limits the number of classes to test against, by default = 10)
       :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
       :param max_iter: maximum number of iterations for deepfool (default = 50)
       :return: minimal perturbation that fools the classifier, number of iterations that it required, new estimated_label and perturbed image
    """
    is_cuda = torch.cuda.is_available()

    if is_cuda:

        print("Using GPU")

        image = image.cuda()

        net = net.cuda()

    else:

        print("Using CPU")

    f_image = net.forward(Variable(image[None, :, :, :], requires_grad=True)).data.cpu().numpy().flatten()

    I = f_image.argsort()[::-1] #将f_image内数据从大到小排列，以下标号的形式返回

    I = I[0:num_classes]

    label = I[0]

    input_shape = image.cpu().numpy().shape

    pert_image = copy.deepcopy(image)

    w = np.zeros(input_shape)

    r_tot = np.zeros(input_shape)

    loop_i = 0

    x = Variable(pert_image[None, :], requires_grad=True)

    fs = net.forward(x)

    k_i = label

    while k_i == label and loop_i < max_iter:

        pert = np.inf

        fs[0, I[0]].backward(retain_graph=True)

        grad_orig = x.grad.data.cpu().numpy().copy()

        for k in range(1, num_classes):

            zero_gradients(x)

            fs[0, I[k]].backward(retain_graph=True)

            cur_grad = x.grad.data.cpu().numpy().copy()

            # set new w_k and new f_k
            w_k = cur_grad - grad_orig

            f_k = (fs[0, I[k]] - fs[0, I[0]]).data.cpu().numpy()

            pert_k = abs(f_k)/np.linalg.norm(w_k.flatten())

            # determine which w_k to use
            if pert_k < pert:

                pert = pert_k

                w = w_k

        # compute r_i and r_tot
        # Added 1e-4 for numerical stability
        r_i = (pert+1e-4) * w / np.linalg.norm(w)

        r_tot = np.float32(r_tot + r_i)

        if is_cuda:

            pert_image = image + (1+overshoot)*torch.from_numpy(r_tot).cuda()

        else:

            pert_image = image + (1+overshoot)*torch.from_numpy(r_tot)

        x = Variable(pert_image, requires_grad=True)

        fs = net.forward(x)

        k_i = np.argmax(fs.data.cpu().numpy().flatten())

        loop_i += 1

    return pert_image


def main():

    cnn = CNN()

    cnn.eval()

    cnn.load_state_dict(torch.load('model1.pkl'))

    print("load cnn net.")

    test_dataloader = my_dataset.get_test_data_loader()

    correct = 0

    total = 0

    #输入deepfool的最大迭代次数
    max_iter = int(input("please input max_iter:"))

    for i, (images, labels) in enumerate(test_dataloader):

        images = torch.squeeze(images, dim=1)

        perturbed_data = Variable(deepfool(images, cnn, 248, 0, max_iter))

        perturbed_label = cnn(perturbed_data)

        perturbed_label = perturbed_label.cpu()

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

        if (total % 10 == 0):
            print('Test Accuracy of the model on the %d test images: %f %%' % (total, 100 * correct / total))

    print('Test Accuracy of the model on the %d test images: %f %%' % (total, 100 * correct / total))


if __name__ == '__main__':
    main()

