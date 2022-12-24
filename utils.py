import os
import cv2
import torch
import numpy as np
import torch.nn as nn
import tensorflow as tf
from torch.autograd import Variable
import matplotlib.pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
__all__ = ['Dataload', 'img_recon', 'Conv2d', 'Deconv2d', 'loss_img', 'RGBload', 'img_recon_rgb']


# RGB图像读取，整体使用RGB进行融合并且读出
def RGBload():
    path = "C:/Users/zpf/Desktop/SSdataset/train/"
    src_Pan = path + "PAN/"
    src_Ms = path + "MS/"
    src_real = path + "Ground Truth/"

    # 加载数据
    x_train_list = os.listdir(src_Pan)
    y_test_list = os.listdir(src_Ms)
    real_test_list = os.listdir(src_real)

    Pan = np.empty((len(x_train_list), 3, 256, 256))  # 生成空的数组
    Ms = np.empty((len(y_test_list), 3, 64, 64))  # 生成空的数组
    real = np.empty((len(real_test_list), 3, 256, 256))  # 生成空的数组

    i = 0
    j = 0
    k = 0

    for name in x_train_list:  # Pan仅导入灰度图
        temp = os.path.join(src_Pan, name)
        img = cv2.imread(temp)
        img = np.swapaxes(img, 0, 2)
        img = np.swapaxes(img, 1, 2)
        Pan[i][:, :, :] = img
        i = i + 1
    for name in y_test_list:  # Ms三个分量分开导入并融合
        temp = os.path.join(src_Ms, name)
        img = cv2.imread(temp)
        img = np.swapaxes(img, 0, 2)
        img = np.swapaxes(img, 1, 2)
        Ms[j][:, :, :] = img
        j = j + 1
    for name in real_test_list:  # 导入real数据
        temp = os.path.join(src_real, name)
        img = cv2.imread(temp)
        img = np.swapaxes(img, 0, 2)
        img = np.swapaxes(img, 1, 2)
        real[k][:, :, :] = img
        k = k + 1

    # flat
    Pan = np.array(Pan).astype('float32') / 255.
    Ms = np.array(Ms).astype('float32') / 255.
    real = np.array(real).astype('float32') / 255.

    Pan = torch.tensor(Pan)
    Pan = Variable(Pan, requires_grad=True)
    Ms = torch.tensor(Ms)
    Ms = Variable(Ms, requires_grad=True)
    real = torch.tensor(real)

    return Pan, Ms, real


# HSV图像读取，整合入空数组
def Dataload():
    src_Pan = "C:/Users/zpf/Desktop/selfdata/pan/"
    src_Ms = "C:/Users/zpf/Desktop/selfdata/ms/"
    src_real = "C:/Users/zpf/Desktop/selfdata/real_ground_image/"

    # 加载数据
    x_train_list = os.listdir(src_Pan)
    y_test_list = os.listdir(src_Ms)
    real_test_list = os.listdir(src_real)

    Pan = np.empty((676, 1, 256, 256))  # 生成空的数组
    Ms = np.empty((676, 3, 64, 64))  # 生成空的数组
    real = np.empty((676, 3, 256, 256))  # 生成空的数组

    i = 0
    j = 0
    k = 0

    for name in x_train_list:  # Pan仅导入灰度图
        temp = os.path.join(src_Pan, name)
        img = cv2.imread(temp, cv2.IMREAD_GRAYSCALE)  # 使用cv2读取图像的时候会被自动转换为3通道，所以需要添加cv2.IMREAD_GRAYSCALE，强制读取灰度图
        img = img.reshape(1, 256, 256)
        Pan[i][:, :, :] = img
        i = i + 1
    for name in y_test_list:  # Ms三个分量分开导入并融合
        temp = os.path.join(src_Ms, name)
        temp = cv2.imread(temp)
        ms = Transform_img(temp)
        Ms[j][0:1, :, :] = ms[0]  # Y 灰度
        Ms[j][1:2, :, :] = ms[1]  # U 色度
        Ms[j][2:3, :, :] = ms[2]  # V 色度
        j = j + 1
    for name in real_test_list:  # 导入real数据
        temp = os.path.join(src_real, name)
        temp = cv2.imread(temp)
        Real = Transform_img(temp)
        real[k][0:1, :, :] = Real[0]  # Y 灰度
        real[k][1:2, :, :] = Real[1]  # U 色度
        real[k][2:3, :, :] = Real[2]  # V 色度
        k = k + 1

    # flat
    Pan_data = np.array(Pan).astype('float32') / 255.
    # Pan = Pan.reshape((len(Pan), np.prod(Pan.shape[1:])))  # (4096, 1, 256, 256) -> (4096, 65536)
    Ms_data = np.array(Ms).astype('float32') / 255.
    # Ms = Ms.reshape((len(Ms), np.prod(Ms.shape[1:])))  # (4096, 1, 64, 64) -> (4096, 4096)
    real = np.array(real).astype('float32') / 255.

    # 数据tensor化，并追踪
    Pan_data = torch.tensor(Pan_data)
    Pan_data = Variable(Pan_data, requires_grad=True)
    Ms_data = torch.tensor(Ms_data)
    Ms_data = Variable(Ms_data, requires_grad=True)
    real = torch.tensor(real)

    return Pan_data, Ms_data, real


# rgb图像训练后重现
def img_recon_rgb(img_data):
    img = img_data.detach().cpu().numpy()  # 从cuda中分离出张量并且转为cpu中在转为numpy（不计入反向传递）
    img = np.swapaxes(img, 2, 3)
    imgs = np.swapaxes(img, 3, 1)
    img = imgs[0][:, :, :]
    img = np.array(img).astype(np.float32)*255

    return img


# 图像重现 根据网络生成的图像是YUV色彩空间的，需要转换为RGB
def img_recon(img_data):
    img_data = img_data.detach().cpu().numpy()  # 从cuda中分离出张量并且转为cpu中在转为numpy（不计入反向传递）
    img_data = Transform_img(img_data, inverse=True)
    n = img_data.shape[0]  # 获取第一维度的大小
    imgs = []
    for i in range(n):
        image = img_data[i][:, :, :]  # 提取第一纬度的不同的数据
        imgs.append(image.tolist())

    img = np.array(imgs[0]).astype(np.float32)  # np默认的数据类型是float64，但是cv2支持的是float32
    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)*255  # 测试使用BGR和HSV分别保存图像，看效果

    return img


# Ms图像数据维度转换(n, 256, 256, 3) -> 3个(n, 1, 256, 256)，合并见Dataload的Ms的合并
# Ms图像数据维度转换3个(n, 1, 256, 256) -> (n, 256, 256， 3)，用于输出图像
def Transform_img(img_data, inverse=False, Norm=False):
    if inverse is False:
        img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2HSV)  # RGB转为YUV
        n = img_data.shape[0]

        img_data1 = img_data[:, :, 0:1]
        img_data2 = img_data[:, :, 1:2]
        img_data3 = img_data[:, :, 2:3]

        V = img_data1.reshape(1, n, n)  # 第三维度分量分离 V色度
        U = img_data2.reshape(1, n, n)  # 第三维度分量分离 U色度
        Y = img_data3.reshape(1, n, n)  # 第三维度分量分离 Y灰度

        if Norm is not False:  # 图像数据归一化
            V = V / 255
            U = U / 255
            Y = Y / 255

        return Y, U, V
    else:
        m = img_data.shape[0]  # 获取第一维度的大小
        n = img_data.shape[-1]
        imgs = np.empty((m, n, n, 3))  # 生成空的数组

        for i in range(m):
            V = img_data[i][0:1, :, :].reshape(n, n, 1)
            U = img_data[i][1:2, :, :].reshape(n, n, 1)
            Y = img_data[i][2:3, :, :].reshape(n, n, 1)

            V = V * 255
            U = U * 255
            Y = Y * 255

            imgs[i][:, :, 0:1] = V
            imgs[i][:, :, 1:2] = U
            imgs[i][:, :, 2:3] = Y
        return imgs


# loss绘图
def loss_img(nu, lo, epoch):
    nu = range(0, nu)
    plt.figure()  # 创建窗口
    plt.plot(nu, lo)
    plt.savefig('./Loss_img/{}.png'.format(epoch))


# 定义torch卷积层
class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bn=True, activation='relu', dropout=False):
        super(Conv2d, self).__init__()
        padding = (kernel_size - 1) // 2  # 向下取整
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding)  # 向下取整
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0, affine=True) if bn else None
        self.dropout = nn.Dropout(p=0.5) if dropout else None
        if activation == 'leakyrelu':
            self.activation = nn.LeakyReLU(negative_slope=0.2)
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise ValueError('Not a valid activation, received {}'.format(activation))

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.activation(x)
        return x


# 定义torch反卷积层
class Deconv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bn=False, activation='leakyrelu', dropout=False):
        super(Deconv2d, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0, affine=True) if bn else None
        self.dropout = nn.Dropout(p=0.5) if dropout else None
        if activation == 'leakyrelu':
            self.activation = nn.LeakyReLU(negative_slope=0.2)
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise ValueError('Not a valid activation, received {}'.format(activation))

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.activation(x)
        return x


# ======================================================tensorflow_utils================================================
# 构造可训练参数
def make_var(name, shape, trainable=True):
    return tf.get_variable(name, shape, trainable=trainable)


# 定义卷积层
def conv2d(input_, output_dim, kernel_size, stride, padding="SAME", name="conv2d", biased=False):
    input_dim = input_.get_shape()[-1]
    with tf.variable_scope(name):
        kernel = make_var(name='weights', shape=[kernel_size, kernel_size, input_dim, output_dim])
        output = tf.nn.conv2d(input_, kernel, [1, stride, stride, 1], padding=padding)
        if biased:
            biases = make_var(name='biases', shape=[output_dim])
            output = tf.nn.bias_add(output, biases)
        return output


# 定义空洞卷积层
def atrous_conv2d(input_, output_dim, kernel_size, dilation, padding="SAME", name="atrous_conv2d", biased=False):
    input_dim = input_.get_shape()[-1]
    with tf.variable_scope(name):
        kernel = make_var(name='weights', shape=[kernel_size, kernel_size, input_dim, output_dim])
        output = tf.nn.atrous_conv2d(input_, kernel, dilation, padding=padding)
        if biased:
            biases = make_var(name='biases', shape=[output_dim])
            output = tf.nn.bias_add(output, biases)
        return output


# 定义反卷积层
def deconv2d(input_, output_dim, kernel_size, stride, padding="SAME", name="deconv2d"):
    input_dim = input_.get_shape()[-1]
    input_height = int(input_.get_shape()[1])
    input_width = int(input_.get_shape()[2])
    with tf.variable_scope(name):
        kernel = make_var(name='weights', shape=[kernel_size, kernel_size, output_dim, input_dim])
        output = tf.nn.conv2d_transpose(input_, kernel, [1, input_height * 2, input_width * 2, output_dim],
                                        [1, 2, 2, 1], padding="SAME")
        return output


# 定义batchnorm(批次归一化)层
def batch_norm(input_, name="batch_norm"):
    with tf.variable_scope(name):
        input_dim = input_.get_shape()[-1]
        scale = tf.get_variable("scale", [input_dim],
                                initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32))
        offset = tf.get_variable("offset", [input_dim], initializer=tf.constant_initializer(0.0))
        mean, variance = tf.nn.moments(input_, axes=[1, 2], keep_dims=True)
        epsilon = 1e-5
        inv = tf.rsqrt(variance + epsilon)
        normalized = (input_ - mean) * inv
        output = scale * normalized + offset
        return output


# 定义instance_norm(增加）
def instance_norm(input_, name="instance_norm"):
    with tf.variable_scope(name):
        epsilon = 1e-9

        mean, var = tf.nn.moments(input_, [1, 2], keep_dims=True)

        return tf.div(tf.subtract(input_, mean), tf.sqrt(tf.add(var, epsilon)))


# 定义Layer normalization
def Layernorm(input_, name="layer_norm"):
    with tf.variable_scope(name):
        input_dim = input_.get_shape()[-1]
        results = 0.
        eps = 1e-5

        scale = tf.get_variable("scale", [input_dim],
                                initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32))
        offset = tf.get_variable("offset", [input_dim], initializer=tf.constant_initializer(0.0))
        x_mean = np.mean(input_, axis=(1, 2, 3), keepdims=True)
        x_var = np.var(input_, axis=(1, 2, 3), keepdims=True)
        x_normalized = (input_ - x_mean) / np.sqrt(x_var + eps)
        results = scale * x_normalized + offset
        return results


# 定义relu激活层
def relu(input_, name="relu"):
    return tf.nn.relu(input_, name=name)


# 定义lrelu激活层
def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak * x)


# 定义残差块
def residule_block_33(input_, output_dim, kernel_size=3, stride=1, dilation=2, atrous=True, name="res"):
    if atrous:
        conv2dc0 = atrous_conv2d(input_=input_, output_dim=output_dim, kernel_size=kernel_size, dilation=dilation,
                                 name=(name + '_c0'))
        conv2dc0_norm = batch_norm(input_=conv2dc0, name=(name + '_bn0'))
        conv2dc0_relu = relu(input_=conv2dc0_norm)
        conv2dc1 = atrous_conv2d(input_=conv2dc0_relu, output_dim=output_dim, kernel_size=kernel_size,
                                 dilation=dilation, name=(name + '_c1'))
        conv2dc1_norm = batch_norm(input_=conv2dc1, name=(name + '_bn1'))
    else:
        conv2dc0 = conv2d(input_=input_, output_dim=output_dim, kernel_size=kernel_size, stride=stride,
                          name=(name + '_c0'))
        conv2dc0_norm = batch_norm(input_=conv2dc0, name=(name + '_bn0'))
        conv2dc0_relu = relu(input_=conv2dc0_norm)
        conv2dc1 = conv2d(input_=conv2dc0_relu, output_dim=output_dim, kernel_size=kernel_size, stride=stride,
                          name=(name + '_c1'))
        conv2dc1_norm = batch_norm(input_=conv2dc1, name=(name + '_bn1'))

    add_raw = input_ + conv2dc1_norm
    output = relu(input_=add_raw)
    return output
