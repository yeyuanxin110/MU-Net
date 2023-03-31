import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

def save_tensor_tf(T, path):
    T = T.squeeze(0)
    # T_numpy = torch.tensor(T, dtype=torch.uint8).permute([1, 2, 0]).detach().cpu().numpy()
    T_numpy = torch.tensor(T).squeeze(0).detach().cpu().numpy()
    T_numpy = float2uint8_tf(T_numpy)
    cv.imwrite(path, T_numpy)

def affine_transform_tf(im1, im2, H12, device='cpu'):
    im1_grid = F.affine_grid(H12, im1.size())
    im1_resample = F.grid_sample(im1, im1_grid)
    a = torch.tensor([[[0, 0, 1]]], dtype=torch.float).to(device)
    a = a.repeat(im1.size()[0], 1, 1)
    affine_matrix = torch.cat([H12, a], dim=1)
    inv_affine_matrix = torch.inverse(affine_matrix)
    H21 = inv_affine_matrix[:, 0:2, :]
    im2_grid = F.affine_grid(H21, im2.size())
    im2_resample = F.grid_sample(im2, im2_grid)
    return im1_resample, im2_resample, H21

def float2uint8_tf(M):
    a = np.max(M)
    b = np.min(M)
    M = (M - b) / (a - b) * 255
    M = M.astype(np.uint8)
    return M

def show_registration_result(ref, sen, sen_tran_T):
    sen_tran_T = sen_tran_T.squeeze(0)
    T_numpy = torch.tensor(sen_tran_T, dtype=torch.uint8).squeeze(0).detach().cpu().numpy()
    plt.subplot(1,3,1)
    plt.imshow(ref, cmap='gray')
    plt.title('input ref')
    plt.subplot(1, 3, 2)
    plt.imshow(sen, cmap='gray')
    plt.title('input sen')
    plt.subplot(1, 3, 3)
    plt.imshow(T_numpy, cmap='gray')
    plt.title('output sen correct')
    plt.show()