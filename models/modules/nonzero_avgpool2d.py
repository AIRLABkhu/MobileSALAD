import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Function


class NonzeroAvgPool2dFunction(Function):
    @staticmethod
    def forward(ctx, input_tensor, kernel_size, stride) -> torch.Tensor:
        ctx.save_for_backward(input_tensor)
        ctx.kernel_size = kernel_size
        ctx.stride = stride

        batch_size, num_channels, input_size, _ = input_tensor.shape
        output_size = int(round((input_size - kernel_size + 1) / stride))
        
        ctx.input_size = input_size
        
        # Unfold and reshape input tensor for pooling
        unfolded = F.unfold(input_tensor, kernel_size=kernel_size, stride=stride)
        unfolded_shape = unfolded.shape
        unfolded = unfolded.reshape(batch_size, num_channels, -1, unfolded.size(-1))

        # Compute mask for non-zero elements
        mask = unfolded.ne(0).float()
        mask = mask.sum(dim=2, keepdim=True)
        mask[mask.eq(0)] = torch.inf
        kernels = torch.reciprocal(mask)
        
        ctx.unfolded_shape = unfolded_shape
        ctx.reshaped_shape = unfolded.shape
        ctx.kernels = kernels
        
        # Normalize and sum over the pooling window
        output = (unfolded * kernels).sum(dim=2)
        return output.reshape(batch_size, num_channels, output_size, output_size)
    
    @staticmethod
    def backward(ctx, grad_output):
        input_size = ctx.input_size
        kernel_size = ctx.kernel_size
        stride = ctx.stride
        unfolded_shape = ctx.unfolded_shape
        reshaped_shape = ctx.reshaped_shape
        kernels = ctx.kernels
        
        kernels = kernels.expand(*reshaped_shape).reshape(unfolded_shape)
        kernels = F.fold(kernels, (input_size, input_size), kernel_size=kernel_size, stride=stride)
        
        num_channels = grad_output.size(1)
        tconv_kernels = torch.ones(num_channels, 1, kernel_size, kernel_size, dtype=torch.float, device=grad_output.device)
        output_count = torch.ones_like(grad_output[:1, :1])
        
        grad_output = F.conv_transpose2d(grad_output, weight=tconv_kernels, stride=stride, groups=num_channels)
        output_count = F.conv_transpose2d(output_count, weight=tconv_kernels[:1], stride=stride)
        
        return grad_output * kernels / output_count, None, None
    
    
def nonzero_avgpool2d(input: torch.Tensor, kernel_size: int, stride: int=1) -> torch.Tensor:
    return NonzeroAvgPool2dFunction.apply(input, kernel_size, stride)


class NonzeroAvgPool2d(nn.Module):
    def __init__(self, kernel_size: int, stride: int|None=None):
        super(NonzeroAvgPool2d, self).__init__()
        self.__kernel_size = kernel_size
        self.__stride = stride
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return nonzero_avgpool2d(input=input, kernel_size=self.__kernel_size, stride=self.__stride)
    
    @property
    def kernel_size(self):
        return self.__kernel_size
    
    @property
    def stride(self):
        return self.__stride


if __name__ == '__main__':
    import torch
    from torch import nn
    import torch.nn.functional as F

    realistic_inputs = True
    device = 0

    if realistic_inputs:
        batch_size, input_size = 64, 16
        kernel_size, stride = 6, 1
        num_channels = 768
        masking_ratio = 0.45
    else:
        batch_size, input_size = 1, 4
        kernel_size, stride = 2, 2
        num_channels = 3
        masking_ratio = 0.4
        
    def make_sample():
        torch.manual_seed(2)
            
        s = torch.arange(batch_size * num_channels * input_size * input_size)
        s = s.reshape(batch_size, num_channels, input_size, input_size).float()

        mask = torch.rand_like(s.float()).mean(dim=1, keepdim=True) > masking_ratio
        s = s * mask
        s = s.cuda(device).requires_grad_(True)

        net = nn.Conv2d(num_channels, num_channels, 1).cuda(device)
        return s, net(s)

    def aggregation_fn(out: torch.Tensor) -> torch.Tensor:
        return (out.square() + 1).log().sum()
        
    def naive_nonzero_avgpool2d(x, kernel_size, stride):
        x = F.unfold(x, kernel_size=kernel_size, stride=stride)
        x = x.reshape(-1, num_channels, kernel_size * kernel_size, x.size(-1))

        mask = x.ne(0).float()
        mask = mask.sum(dim=2, keepdim=True)
        mask[mask.eq(0)] = torch.inf

        x = (x / mask).sum(dim=2)
        output_size = int(round((input_size - kernel_size + 1) / stride))
        x = x.reshape(-1, num_channels, output_size, output_size)
        return x

    t1, s1 = make_sample()
    out1 = naive_nonzero_avgpool2d(s1, kernel_size=kernel_size, stride=stride)
    aggregation_fn(out1).backward()
    grad1 = t1.grad
        
    t2, s2 = make_sample()
    out2 = nonzero_avgpool2d(s2, kernel_size, stride)
    aggregation_fn(out2).backward()
    grad2 = t2.grad
    
    assert torch.allclose(out1, out2)
    assert torch.allclose(grad1, grad2)
    print('PASSED:', __file__)
    