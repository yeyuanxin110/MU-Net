from tool.CFOG import CFOG
import torch

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

def ComputeLoss(reference, sensed_tran, sensed, reference_inv_tran):
    loss_1 = NCC_loss(reference, sensed_tran)
    loss_2 = NCC_loss(sensed, reference_inv_tran)
    loss = loss_1 + loss_2
    return loss

def gncc_loss(I, J, eps=1e-5):
    I2 = I.pow(2)
    J2 = J.pow(2)
    IJ = I*J
    I_ave, J_ave = I.mean(), J.mean()
    I2_ave, J2_ave = I2.mean(), J2.mean()
    IJ_ave = IJ.mean()
    cross = IJ_ave - I_ave * J_ave
    I_var = I2_ave - I_ave.pow(2)
    J_var = J2_ave - J_ave.pow(2)
    cc = cross / (I_var.sqrt() * J_var.sqrt() + eps)  # 1e-5
    return -1.0 * cc + 1