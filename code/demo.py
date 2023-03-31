from __future__ import print_function
import torch
import torch.optim as optim
import numpy as np
from PIL import Image
from tool.loss_tools import ComputeLoss
from tool.model_tools import net, check_model_path
from tool.preprocess_tools import affine_transform_tf, show_registration_result, save_tensor_tf
import os
from torch.autograd import Variable
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import warnings
warnings.filterwarnings('ignore')
import time

def ite(ref_img, sen_img, pretrained_model=None):
    time_start = time.time()
    model = net().to(device)
    model = check_model_path(model, pretrained_model)
    print('Using device: ' + str(device))
    print('Registration Waiting...')
    save_sen_tran_name = 'save.jpg'
    parameter_learn_rate = 0.0004
    max_iter = 800
    stop_iter = 200
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
        affine_parameter = model(torch.cat([ref_tensor, sen_tensor], dim=1))
        sen_tran_tensor, ref_inv_tensor, inv_affine_parameter = affine_transform_tf(sen_tensor, ref_tensor,
                                                                                affine_parameter, device)
        loss = ComputeLoss(ref_tensor, sen_tran_tensor, sen_tensor, ref_inv_tensor)
        Loss.append(loss)
        loss.backward()
        if loss < loss_0 and torch.isnan(loss).any() == False:
            count = 0
            sen_tran_tensor_save = sen_tran_tensor
            save_tensor_tf(sen_tran_tensor_save, save_sen_tran_name)
            loss_0 = loss
            if epoch>100:
                parameter_learn_rate = parameter_learn_rate*0.975
                optimizer = optim.SGD(model.parameters(), lr=parameter_learn_rate)

        if count > stop_iter:
            break
        optimizer.step()
    time_end = time.time()
    print('time cost', time_end - time_start, 's')
    show_registration_result(ref_img, sen_img, sen_tran_tensor_save)

if __name__ == "__main__":
    ref_img = Image.open('../data/Optical-SAR/1_9_1.jpg')
    sen_img = Image.open('../data/Optical-SAR/1_9_3.jpg')
    ite(ref_img, sen_img)