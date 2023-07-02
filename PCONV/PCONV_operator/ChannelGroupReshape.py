import torch
import torch.nn as nn
import PCONV
from PCONV_operator.BaseOpModule import BaseOpModule

class ChannelGroupReshape_AF(torch.autograd.Function):

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
    

class ChannelGroupReshape(BaseOpModule):
    
    def __init__(self, ngroup, code_channels, device = 0, time_it = False):
        super(ChannelGroupReshape, self).__init__(device)
        self.op = { gid : PCONV.ChannelGroupReshapeOp(ngroup,code_channels, gid, time_it) for gid in self.device_list}
        

    def forward(self, x):
        res = ChannelGroupReshape_AF.apply(x, self.op)
        return res


class ChannelGroupReshapeUneven_AF(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, x, param, op):
        gid = x.device.index
        outputs = op[gid].forward_uneven(x, param)
        ctx.save_for_backward(param)
        ctx.op = op
        return outputs[0]
        
    @staticmethod
    def backward(ctx, grad_output):
        gid = grad_output.device.index
        param = ctx.saved_tensors[0]
        outputs = ctx.op[gid].backward_uneven(grad_output,param)
        return outputs[0], None, None
    

class ChannelGroupReshapeUneven(BaseOpModule):
    
    def __init__(self, code_channels, mode=0, device = 0, time_it = False):
        super(ChannelGroupReshapeUneven, self).__init__(device)
        ngroup = 5 if mode == 0 else (10 if mode == 1 else 20)
        self.op = { gid : PCONV.ChannelGroupReshapeOp(ngroup,code_channels, gid, time_it) for gid in self.device_list}
        if mode == 0:
            pm = [0, 0.05, 0.1, 0.2, 0.4, 1]
        elif mode == 1:
            pm = [0., 0.025, 0.05, 0.1, 0.2, 0.5, 0.525, 0.55, 0.6, 0.7, 1]
        else:
            pm = [0.0, 0.0125, 0.025, 0.05, 0.1, 0.25, 0.2625, 0.275, 0.3, 0.35, 0.5, 0.5125, 0.525, 0.55, 0.6, 0.75, 0.7625, 0.775, 0.8, 0.85, 1]
        pm = torch.Tensor(pm).to(device='cuda:{}'.format(device),dtype=torch.float32)
        self.register_buffer('param',torch.floor(pm*code_channels+0.5).type(torch.int32))
        

    def forward(self, x):
        res = ChannelGroupReshapeUneven_AF.apply(x, self.param, self.op)
        return res

def mimic_forward_uneven(x,code_channels):
    pm = [int(pt*code_channels)for pt in[0, 0.05, 0.1, 0.2, 0.4, 1]]
    ngroup = len(pm) - 1
    ch_list = [0 for _ in range(code_channels)]
    for pc in range(code_channels):
        for idx in range(ngroup):
            if pc < pm[idx+1]:
                ch_list[pc] = idx
                break
    n,c,h,w = x.shape
    cpg = c // ngroup
    y = torch.zeros((n,code_channels,h,w,cpg),dtype=torch.float32).type_as(x)
    for ci in range(cpg):
        for pc in range(code_channels):
            pg = ch_list[pc]
            y[:,pc,:,:,ci] = x[:,pg*cpg+ci,:,:]
    return y

def test_uneven():
    import math
    n,c,h,w = 2, 80, 3, 3
    data = torch.autograd.Variable(torch.rand((n,c,h,w),dtype=torch.float32,device='cuda:0'),requires_grad = True)
    data.retain_grad()
    cgr = ChannelGroupReshapeUneven(40,1,0).to('cuda:0')
    y = cgr(data)
    loss = torch.sum(y**2/2)
    loss.backward()
    '''
    grad = data.grad.clone().detach()
    delta = 0.2
    for ni in range(n):
        for ci in range(c):
            for hi in range(h):
                for wi in range(w):
                    data.data[ni,ci,hi,wi] += delta
                    y = cgr(data)
                    l1 = torch.sum(y**2/2).item()
                    data.data[ni,ci,hi,wi] -= 2*delta
                    y = cgr(data)
                    l2 = torch.sum(y**2/2).item()
                    data.data[ni,ci,hi,wi] += delta
                    g1, g2 = (l1-l2)/2/delta , grad[ni,ci,hi,wi].item()
                    if math.fabs(g1-g2) / (math.fabs(g1)+1e-7) > 1e-3:
                        print(ni,ci,hi,wi, g1, g2)
    '''

def mimic_forward(x,code_channels,ngroup):
    
    n,c,h,w = x.shape
    cpg = c // ngroup
    ct = code_channels // ngroup
    y = torch.zeros((n,code_channels,h,w,cpg),dtype=torch.float32).type_as(x)
    for ci in range(cpg):
        for pc in range(code_channels):
            pg = pc // ct
            y[:,pc,:,:,ci] = x[:,pg*cpg+ci,:,:]
    return y

def test_even():
    import math
    n,c,h,w = 2, 8, 2, 2
    code_channels,ngroup = 12, 4
    data = torch.autograd.Variable(torch.arange(1,n*c*h*w+1,dtype=torch.float32,device='cuda:0').view(n,c,h,w),requires_grad = True)
    #data = torch.autograd.Variable(torch.rand((n,c,h,w),dtype=torch.float32,device='cuda:0'),requires_grad = True)
    data.retain_grad()
    cgr = ChannelGroupReshape(ngroup,code_channels,0).to('cuda:0')
    y = cgr(data)
    y2 = mimic_forward(data,code_channels,ngroup)
    print(torch.sum(torch.abs(y2.view(-1)-y.view(-1))))
    test_grad = False
    if test_grad:
        loss = torch.sum(y**2/2)
        loss.backward()
        grad = data.grad.clone().detach()
        delta = 0.3
        for ni in range(n):
            for ci in range(c):
                for hi in range(h):
                    for wi in range(w):
                        data.data[ni,ci,hi,wi] += delta
                        y = cgr(data)
                        l1 = torch.sum(y**2/2).item()
                        data.data[ni,ci,hi,wi] -= 2*delta
                        y = cgr(data)
                        l2 = torch.sum(y**2/2).item()
                        data.data[ni,ci,hi,wi] += delta
                        g1, g2 = (l1-l2)/2/delta , grad[ni,ci,hi,wi].item()
                        if math.fabs(g1-g2) / (math.fabs(g1)+1e-7) > 1e-1:
                            print(ni,ci,hi,wi, g1, g2)

if __name__ == '__main__':
    test_uneven()




