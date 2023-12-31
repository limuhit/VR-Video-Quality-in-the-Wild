import torch
import torch.nn as nn
import PCONV
from PCONV_operator.BaseOpModule import BaseOpModule

class DTOW_AF(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, op):
        if not x.is_contiguous(): x = x.contiguous()
        gid = x.device.index
        outputs = op[gid].forward(x)
        ctx.op = op
        return outputs[0]
        
    @staticmethod
    def backward(ctx, grad_output):
        if not grad_output.is_contiguous(): grad_output = grad_output.contiguous()
        gid = grad_output.device.index
        outputs = ctx.op[gid].backward(grad_output)
        return outputs[0], None
    

class Dtow(BaseOpModule):

    def __init__(self, stride=2, d2w = False, version=0, device = 0, time_it = False):
        super(Dtow, self).__init__(device)
        self.op = {gid:PCONV.DtowOp(stride, d2w, version, gid, time_it) for gid in self.device_list}
    


    def forward(self, x):
        res = DTOW_AF.apply(x, self.op)
        return res
