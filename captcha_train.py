# -*- coding: UTF-8 -*-

import torch

import torch.nn as nn

from torch.autograd import Variable

import my_dataset

from captcha_cnn_model import CNN


if_use_gpu = 1
# Hyper Parameters

num_epochs = 100

batch_size = 100

learning_rate = 0.001


def main():

    cnn = CNN()

    if if_use_gpu:
        cnn = cnn.cuda()

    cnn.train()

    print('init net')

    criterion = nn.MultiLabelSoftMarginLoss()

    optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)

    # Train the Model
    train_dataloader = my_dataset.get_train_data_loader()

    for epoch in range(num_epochs):

        for i, (images, labels) in enumerate(train_dataloader):

            images = Variable(images)

            labels = Variable(labels.float())

            if if_use_gpu:

                images = images.cuda()

                labels = labels.cuda()

            predict_labels = cnn(images)

            loss = criterion(predict_labels, labels)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            if (i+1) % 10 == 0:

                print("epoch:", epoch, "step:", i, "loss:", loss.item())

            if (i+1) % 100 == 0:

                torch.save(cnn.state_dict(), "./model1.pkl")   #current is model.pkl

                print("save model1")

        print("epoch:", epoch, "step:", i, "loss:", loss.item())

    torch.save(cnn.state_dict(), "./model1.pkl")   #current is model.pkl

    print("save last model")


if __name__ == '__main__':

    main()