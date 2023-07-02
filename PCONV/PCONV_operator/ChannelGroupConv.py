import torch
import torch.nn as nn
import PCONV
from PCONV_operator.BaseOpModule import BaseOpModule

class ChannelGroupParam_AF(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, op):
        gid = x.device.index
        outputs = op[gid].forward(x)
        ctx.op = op
        return outputs[0]
        
    @staticmethod
    def backward(ctx, grad_output):
        gid = grad_output.device.index
        outputs = ctx.op[gid].backward(grad_output)
        return outputs[0], None
    
class ChannelGroupParam(BaseOpModule):
    
    def __init__(self, ngroup, cin, cout, kernel_size=3, hidden=True, device = 0, time_it = False):
        super(ChannelGroupParam, self).__init__(device)
        self.op = { gid : PCONV.ChannelGroupParamOp(ngroup,cin,cout,hidden,gid,time_it) for gid in self.device_list}
        self.weight = nn.Parameter(torch.zeros((cout*ngroup,cin*ngroup,kernel_size,kernel_size),device='cuda:{}'.format(device),dtype=torch.float32))
        nn.init.kaiming_uniform_(self.weight.data)


    def forward(self):
        res = ChannelGroupParam_AF.apply(self.weight, self.op)
        return res

class ChannelGroupConv(nn.Module):

    def __init__(self,ngroup,cin,cout,kernel_size,hidden=True, device_id = 0):
        super(ChannelGroupConv,self).__init__()
        self.weight = ChannelGroupParam(ngroup,cin,cout,kernel_size,hidden,device_id)
        self.bias = nn.Parameter(torch.zeros(cout*ngroup,dtype=torch.float32))

    def forward(self,x):
        wt = self.weight()
        res = nn.functional.conv2d(x,wt,self.bias)
        return res


if __name__ == '__main__':
    data = torch.autograd.Variable(torch.rand((2,20,10,10),dtype=torch.float32,device='cuda:0'),requires_grad = True)
    data.retain_grad()
    cgconv = ChannelGroupConv(5,4,3,False).to('cuda:0')
    y = cgconv(data)
    loss = torch.sum(y**2/2)
    loss.backward()
    pass
