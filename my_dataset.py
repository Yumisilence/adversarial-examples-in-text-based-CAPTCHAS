# -*- coding: UTF-8 -*-

import os

from torch.utils.data import DataLoader,Dataset

import torchvision.transforms as transforms

from PIL import Image

import one_hot_encoding as ohe

import captcha_setting


class mydataset(Dataset):

    def __init__(self, folder, transform=None):

        self.train_image_file_paths = [os.path.join(folder, image_file) for image_file in os.listdir(folder)]

        self.transform = transform


    def __len__(self):  # 返回数据集的大小

        return len(self.train_image_file_paths)



    def __getitem__(self, idx): #获取第i个样本(0索引)

        image_root = self.train_image_file_paths[idx]

        image_name = image_root.split(os.path.sep)[-1]

        image = Image.open(image_root)

        if self.transform is not None:

            image = self.transform(image)

        label = ohe.encode(image_name.split('_')[0]) # 为了方便，在生成图片的时候，图片文件的命名格式 "4个数字或者数字_时间戳.PNG", 4个字母或者即是图片的验证码的值，字母大写,同时对该值做 one-hot 处理

        return image, label


transform = transforms.Compose([

    transforms.Grayscale(),

    transforms.ToTensor(),

])


def get_train_data_loader():

    dataset = mydataset(captcha_setting.TRAIN_DATASET_PATH, transform=transform)

    return DataLoader(dataset, batch_size=64, shuffle=True) #1.批量读取数据 2.打乱数据顺序 3.使用multiprocessing并行加载数据


def get_test_data_loader():

    dataset = mydataset(captcha_setting.TEST_DATASET_PATH, transform=transform)

    return DataLoader(dataset, batch_size=1, shuffle=True)


def get_predict_data_loader():

    dataset = mydataset(captcha_setting.PREDICT_DATASET_PATH, transform=transform)

    return DataLoader(dataset, batch_size=1, shuffle=True)

