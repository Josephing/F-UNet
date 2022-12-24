import os
import cv2
import numpy as np
import torch.nn
from torch.cuda.amp import autocast, GradScaler

from UNet_Fusion_Net import *
from torchsummary import summary
from torch.autograd import Variable

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

eps = 1e-08


def RGBload():
    # global diff

    src_Pan = "C:/Users/Administrator.SC-202008062009/Desktop/SSdataset/train/train2/PAN/"
    src_Ms = "C:/Users/Administrator.SC-202008062009/Desktop/SSdataset/train/train2/MS/"
    src_real = "C:/Users/Administrator.SC-202008062009/Desktop/SSdataset/train/train2/Ground Truth/"

    # 加载数据
    x_train_list = os.listdir(src_Pan)
    y_test_list = os.listdir(src_Ms)
    label_list = os.listdir(src_real)

    Pan = np.empty((len(x_train_list), 3, 256, 256))  # 生成空的数组
    Ms = np.empty((len(y_test_list), 3, 64, 64))  # 生成空的数组
    label = np.empty((len(label_list), 3, 256, 256))  # 生成空的数组

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
    for name in label_list:  # label三个分量分开导入并融合
        temp = os.path.join(src_real, name)
        img = cv2.imread(temp)
        img = np.swapaxes(img, 0, 2)
        img = np.swapaxes(img, 1, 2)
        label[k][:, :, :] = img
        k = k + 1

    # flat

    Pan = np.array(Pan).astype('float16') / 255  # float32
    Ms = np.array(Ms).astype('float16') / 255
    label = np.array(label).astype('float16') / 255

    # # normalize
    # max_patch, min_patch = np.max(Ms, axis=(0, 1)), np.min(Ms, axis=(0, 1))
    # Ms = np.float32((Ms - min_patch) / (max_patch - min_patch))
    # max_patch, min_patch = np.max(Pan, axis=(0, 1)), np.min(Pan, axis=(0, 1))
    # Pan = np.float32((Pan - min_patch) / (max_patch - min_patch))
    # max_patch, min_patch = np.max(label, axis=(0, 1)), np.min(label, axis=(0, 1))
    # Label = np.float32((label - min_patch) / (max_patch - min_patch))
    #
    # diff = max_patch - min_patch
    # print(diff)

    Pan = torch.tensor(Pan)
    Pan = Variable(Pan, requires_grad=True)
    Ms = torch.tensor(Ms)
    Ms = Variable(Ms, requires_grad=True)
    Label = torch.tensor(label)
    Label = Variable(Label, requires_grad=True)

    return Pan, Ms, Label


class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, x, y):
        mse_loss = torch.mean(torch.pow((x - y), 2))
        mae_loss = torch.mean(torch.abs(x - y))
        return 0.9 * mse_loss + 0.01 * (1 - ssim(x, y, 11, 'mean', 1.)) + 0.09 * mae_loss


if __name__ == '__main__':

    min_loss = 100
    Epoch = 1000
    Batch = 388 // 1

    loss_list = []

    data = RGBload()

    model = model_rgb()
    model.cuda()

    # summary(model.cuda(), input_size=[(3, 256, 256), (3, 64, 64)])

    # Loss = CustomLoss()
    Loss = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    model.load_state_dict(torch.load('./UNet_weights/UNet_100.pkl'))

    scaler = GradScaler()
    for epoch in range(Epoch):
        for i in range(Batch):
            pan_data = data[0][i:i + 1, :, :, :].cuda()
            ms_data = data[1][i:i + 1, :, :, :].cuda()
            real_data = data[2][i:i + 1, :, :, :].cuda()

            optimizer.zero_grad()

            with autocast():  # 加速训练同时防止溢出错误（nan错误）
                G_data = model(pan_data, ms_data)
                loss = Loss(G_data, real_data)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5, norm_type=2)  # 梯度裁剪
            scaler.step(optimizer)
            scaler.update()
            l_loss = loss.item()
            loss_list.append(l_loss)

            print('epoch {:d} step {:d} \t loss_X = {:.6f}'.format(epoch, i, l_loss))

            if i in range(0, Batch, 2):
                image_recon = img_recon_rgb(G_data)
                cv2.imwrite("./images_G/{}_{}.png".format(epoch, i), image_recon)

        loss_img(Batch, loss_list, epoch)

        temp = 0
        for item in loss_list:
            temp += item
        avg_loss = temp / len(loss_list)

        if min_loss >= avg_loss:
            min_loss = avg_loss
            print("the min_loss=", min_loss)
            torch.save(model.state_dict(), './UNet_weights/best_UNet_{}.pkl'.format(epoch))

        if epoch in range(0, Epoch, 50) and epoch != 0:
            torch.save(model.state_dict(), './UNet_weights/UNet_{}.pkl'.format(epoch))

        loss_list.clear()
