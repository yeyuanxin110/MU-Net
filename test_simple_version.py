from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from PIL import Image
from torch import inverse
from losses import CFOG, gncc_loss
import os
import matplotlib.pyplot as plt
from torch.autograd import Variable
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import warnings
warnings.filterwarnings('ignore')
import time


def NCC_loss(i, j):
    x = torch.ge(i.squeeze(0).squeeze(0), 1)
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.ge(j.squeeze(0).squeeze(0), 1)
    y = torch.tensor(y, dtype=torch.float32)
    z = torch.mul(x, y)
    num = z[z.ge(1)].size()[0]
    CFOG_sar = torch.mul(CFOG(i), z)
    CFOG_optical = torch.mul(CFOG(j), z)
    loss = gncc_loss(CFOG_sar, CFOG_optical)*10000/num
    return loss


def save_tensor_to_image(T, path):
    T = T.squeeze(0)
    # T_numpy = torch.tensor(T, dtype=torch.uint8).permute([1, 2, 0]).detach().cpu().numpy()
    T_numpy = torch.tensor(T, dtype=torch.uint8).squeeze(0).detach().cpu().numpy()
    T_PIL = Image.fromarray(T_numpy)
    T_PIL.save(path)


class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding, dilation=1):
        super(ConvBlock, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size,
                                stride=stride, padding=padding, dilation=dilation)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(True)
    def forward(self, x):
        out = self.conv2d(x)
        out = self.bn(out)
        out = self.relu(out)
        return out


class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.localization = nn.Sequential(
            ConvBlock(2, 32, 3, 2, 1, 3),
            ConvBlock(32, 32, 3, 2, 1, 3),
            ConvBlock(32, 32, 3, 2, 1, 3),
            ConvBlock(32, 32, 3, 2, 1, 3),
            ConvBlock(32, 32, 3, 2, 1, 1),
            ConvBlock(32, 32, 3, 2, 1, 1)
        )
        self.fc = nn.Conv2d(32, 6, 8, 1, 0)
        self.fc.weight.data.zero_()
        self.fc.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x, y):
        c = torch.cat([x, y], dim=1)
        c = self.localization(c)
        c = self.fc(c)
        c = c.view(-1, 2, 3)
        return c


def AffineTransform(reference, sensed, affine_matrix):
    sensed_grid = F.affine_grid(affine_matrix, sensed.size())
    sensed_tran = F.grid_sample(sensed, sensed_grid)
    a = torch.tensor([[[0, 0, 1]]], dtype=torch.float).to(device)
    affine_matrix = torch.cat([affine_matrix, a], dim=1)
    inv_affine_matrix = inverse(affine_matrix)
    inv_affine_matrix = inv_affine_matrix[:, 0:2, :]
    reference_grid = F.affine_grid(inv_affine_matrix, reference.size())
    reference_inv_tran = F.grid_sample(reference, reference_grid)
    return sensed_tran, reference_inv_tran, inv_affine_matrix


def ComputeLoss(reference, sensed_tran, sensed, reference_inv_tran):
    loss_1 = NCC_loss(reference, sensed_tran)
    loss_2 = NCC_loss(sensed, reference_inv_tran)
    loss = loss_1 + loss_2
    return loss

def show_plot(iteration, loss, name):
    plt.plot(iteration, loss)
    plt.savefig('./%s' % name)
    plt.show()

def show_tensor(ref, sen, sen_tran_T):
    sen_tran_T = sen_tran_T.squeeze(0)
    # T_numpy = torch.tensor(T, dtype=torch.uint8).permute([1, 2, 0]).detach().cpu().numpy()
    T_numpy = torch.tensor(sen_tran_T, dtype=torch.uint8).squeeze(0).detach().cpu().numpy()
    plt.subplot(1,3,1)
    plt.imshow(ref)
    plt.title('input ref')
    plt.subplot(1, 3, 2)
    plt.imshow(sen)
    plt.title('input sen')
    plt.subplot(1, 3, 3)
    plt.imshow(T_numpy)
    plt.title('output sen correct')
    plt.show()

def ite(ref_img, sen_img, pretrained_model=None):
    time_start = time.time()
    model = torch.load(pretrained_model) \
        if pretrained_model else SimpleNet().to(device)
    print('Using device: ' + str(device))
    print('waiting...')
    save_sen_tran_name = 'save.jpg'
    parameter_learn_rate = 0.001
    max_iter = 3000
    stop_iter = 150
    ref_tensor = Variable(torch.tensor(np.float32(np.array(ref_img))).to(device)).unsqueeze(0).unsqueeze(0)
    sen_tensor = Variable(torch.tensor(np.float32(np.array(sen_img))).to(device)).unsqueeze(0).unsqueeze(0)
    optimizer = optim.SGD(model.parameters(), lr=parameter_learn_rate)
    model.train()
    Epoch = []
    Loss = []
    loss_0 = 1000000
    count = 0
    for epoch in range(max_iter):
        count = count + 1
        Epoch.append(epoch)
        optimizer.zero_grad()
        affine_parameter = model(ref_tensor, sen_tensor)
        sen_tran_tensor, ref_inv_tensor, inv_affine_parameter = AffineTransform(ref_tensor, sen_tensor,
                                                                                affine_parameter)
        loss = ComputeLoss(ref_tensor, sen_tran_tensor, sen_tensor, ref_inv_tensor)
        Loss.append(loss)
        loss.backward()
        if loss < loss_0 and torch.isnan(loss).any() == False:
            count = 0
            save_tensor_to_image(sen_tran_tensor, save_sen_tran_name)
            loss_0 = loss
            if epoch>100:
                parameter_learn_rate = parameter_learn_rate*0.975
                optimizer = optim.SGD(model.parameters(), lr=parameter_learn_rate)

        if count > stop_iter:
            break
        optimizer.step()
    time_end = time.time()
    print('time cost', time_end - time_start, 's')
    show_tensor(ref_img, sen_img, sen_tran_tensor)

if __name__ == "__main__":
    pretrained_model = None # The test demo could be run without pretrained model within iteration
    ref_img = Image.open('./test_images/1_9_1.jpg')
    sen_img = Image.open('./test_images/1_9_3.jpg')
    ite(ref_img, sen_img, pretrained_model)
