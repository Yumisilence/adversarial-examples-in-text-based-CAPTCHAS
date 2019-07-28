# Adversarial Examples in text-based CAPTCHAS

本项目以Pytorch框架为基础，主要实现两部分功能。<br>
第一部分为：程序生成四字文本验证码，使用CNN网络训练分类器模型。<br>
第二部分为：分别使用FGSM算法、I-FGSM算法、Deepfool算法生成对抗性文本验证码，降低第一部分生成的分类器模型的识别准确率。<br>

## 训练分类器模型

`captcha_setting.py` ：设置文本验证码由26个字母大小写和数字组成，字符个数为4，图像尺寸大小为160*60，以及训练集/测试集/预测集文件夹保存位置。<br>
`my_dataset.py`：对训练集/测试集/预测集的数据进行DataLoader设置。<br>
`captcha_gen.py`：程序随机产生文本验证码。<br>
`one_hot_encoding.py`：对每个字符进行one-hot编码，将验证码字符各类别变量转换为CNN更容易利用的一种形式，每个字符用62位二进制表示。<br>
`captcha_cnn_model.py`：含有5层卷积的cnn模型。<br>
`captcha_train.py`：分类器模型训练。<br>
`captcha_predict.py`：使用模型进行文本验证码字符预测<br>
`captcha_test.py`：模型结果测试。<br>


## 生成对抗样本文本验证码

`fgsm_attact.py`:用FGSM算法生成对抗样本。<br>
`i_fgsm_attack.py`：用I-FGSM算法生成对抗样本。<br>
`deepfool_attack.py`：用Deepfool算法生成对抗样本。<br>
