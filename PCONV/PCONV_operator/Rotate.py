import torch
import torch.nn as nn
import PCONV
from PCONV_operator.BaseOpModule import BaseOpModule
import numpy as np

class Rotate_AF(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, theta_phi, op):
        x = x.contiguous()
        theta_phi = theta_phi.contiguous()
        gid = x.device.index
        outputs = op[gid].forward(x, theta_phi)
        ctx.op = op
        return outputs[0]
        
    @staticmethod
    def backward(ctx, grad_output):
        #gid = grad_output.device.index
        #outputs = ctx.op[gid].backward(grad_output)
        return None, None, None
    

class Rotate(BaseOpModule):
    
    def __init__(self, interp=0, device = 0, time_it = False):
        super(Rotate, self).__init__(device)
        self.op = { gid : PCONV.RotateOp(interp, gid, time_it) for gid in self.device_list}
        self.pi = np.pi
        

    def forward(self, x, theta_phi):
        theta_phi = theta_phi / 180 * self.pi
        res = Rotate_AF.apply(x, theta_phi, self.op)
        return res


def tensor2img(x):
    tx = x.to('cpu').detach().numpy().transpose(1,2,0)
    tx[tx<0]=0
    tx[tx>255]=255
    tx = tx.astype(np.uint8)
    return tx

if __name__ == '__main__':
    import cv2
    img_name = 'e:/360_dataset/360_512/44068324_c63bfd1ee7_k.png'
    img = cv2.imread(img_name)
    data = torch.from_numpy(img.transpose(2,0,1).astype(np.float32)).view(1,3,512,1024).contiguous().to('cuda:0')
    vp = Rotate().to('cuda:0')
    tf = torch.Tensor([180,0]).to('cuda:0').contiguous().view(1,2)
    y = vp(data,tf)
    dst = tensor2img(y[0])
    #print(vp.rota)
    cv2.imshow('dst',dst)
    cv2.imshow('src',img)
    cv2.waitKey()