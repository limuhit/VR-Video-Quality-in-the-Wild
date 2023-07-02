import torch
import torch.nn as nn
import PCONV
from PCONV_operator.BaseOpModule import BaseOpModule

class ChannelGroupParamUneven_AF(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, x, param, op):
        gid = x.device.index
        outputs = op[gid].forward_uneven(x,param)
        ctx.save_for_backward(param)
        ctx.op = op
        return outputs[0]
        
    @staticmethod
    def backward(ctx, grad_output):
        gid = grad_output.device.index
        param = ctx.saved_tensors[0]
        outputs = ctx.op[gid].backward_uneven(grad_output,param)
        return outputs[0], None, None

class ChannelGroupUnevenParam(BaseOpModule):
    
    def __init__(self, ngroup, channels, cout, kernel_size=3, mode=0, device = 0, time_it = False):
        super(ChannelGroupUnevenParam, self).__init__(device)
        self.op = { gid : PCONV.ChannelGroupParamOp(ngroup,1,cout,False,gid,time_it) for gid in self.device_list}
        self.weight = nn.Parameter(torch.zeros((cout*ngroup,channels,kernel_size,kernel_size),device='cuda:{}'.format(device),dtype=torch.float32))
        if mode == 0:
            pm = [0, 0.05, 0.1, 0.2, 0.4]
        elif mode == 1:
            pm = [0., 0.025, 0.05, 0.1, 0.2, 0.5, 0.525, 0.55, 0.6, 0.7]
        else:
            pm = [0.0, 0.0125, 0.025, 0.05, 0.1, 0.25, 0.2625, 0.275, 0.3, 0.35, 0.5, 0.5125, 0.525, 0.55, 0.6, 0.75, 0.7625, 0.775, 0.8, 0.85]
        pm = torch.Tensor(pm).to(device='cuda:{}'.format(device),dtype=torch.float32)
        self.register_buffer('param',torch.floor(pm*channels+0.5).type(torch.int32))
        nn.init.kaiming_uniform_(self.weight.data)


    def forward(self):
        res = ChannelGroupParamUneven_AF.apply(self.weight, self.param, self.op)
        return res

class ChannelGroupConvUneven(nn.Module):
    
    def __init__(self, channels,cout,kernel_size, mode=0, device_id = 0):
        super(ChannelGroupConvUneven,self).__init__()
        ngroup = 5 if mode == 0 else (10 if mode == 1 else 20)
        self.weight = ChannelGroupUnevenParam(ngroup,channels,cout,kernel_size,mode,device_id)
        self.bias = nn.Parameter(torch.zeros(cout*ngroup,dtype=torch.float32))

    def forward(self,x):
        wt = self.weight()
        res = nn.functional.conv2d(x,wt,self.bias)
        return res

if __name__ == '__main__':
    data = torch.autograd.Variable(torch.rand((2,20,10,10),dtype=torch.float32,device='cuda:0'),requires_grad = True)
    data.retain_grad()
    cgconv = ChannelGroupConvUneven(20,4,3,0).to('cuda:0')
    y = cgconv(data)
    loss = torch.sum(y**2/2)
    loss.backward()
    pass
