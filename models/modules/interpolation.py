import torch
from torch import nn
from models.modules import NonzeroAvgPool2d, SpConvAdapter


class Interpolation2d(nn.Module):
    '''
    DESCRIPTION:
    
        Interpolate a sparse feature map through sparse convolution and masked average pooling. 
    
    
    PARAMETERS:
    
        `in_dim`: The feature dimension of the input (required).

        `out_dim`: The desired output feature dimension (required).

        `conv_kernel_size`: The kernel size for convolution step (required).

        `pool_kernel_size`: The kernel size for pooling step (required).

        `conv_stride`: The stride for the convolution step (default: `1`).

        `conv_bias`: Indicates whether to apply bias parameter in convolution step (default: `True`).

        `pool_stride`: The stride for the pooling step (default: `1`).
        
        `nonlinearity`: The type of the non-linear layer follows before convolution step (default: `torch.nn.Identity`).

        `nonlin_kwargs`: The parameter dictionary for the non-linear layer (default: an empty `dict`).
        
    
    INPUTS:
    
        `x`: A tensor of shape (`batch_size`, `size`, `size`, `dim_in`).


    OUTPUTS: 
    
        A tensor of shape (`batch_size`, `pooled_size`, `pooled_size`, `dim_out`).
    '''
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        conv_kernel_size: int,
        pool_kernel_size: int,
        conv_stride: int=1,
        conv_bias: int=True,
        pool_stride: int=1,
        nonlinearity: type=nn.Identity,
        nonlin_kwargs: dict={}):
        super(Interpolation2d, self).__init__()
        
        self.conv = SpConvAdapter(in_dim, out_dim, conv_kernel_size, conv_stride, bias=conv_bias)
        self.nonlin = nonlinearity(**nonlin_kwargs)
        self.pool = NonzeroAvgPool2d(pool_kernel_size, pool_stride)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.nonlin(self.conv(x))
        x = self.pool(x.permute(0, 3, 1, 2))
        return x
    
    
if __name__ == '__main__':
    batch_size, input_size, output_size = 64, 16, 11
    in_dim, out_dim = 768, 384
    
    x = torch.randn(batch_size, input_size, input_size, in_dim)
    x = x.cuda().requires_grad_(True)
    
    pool_kernel_size = 16 - 11 + 1
    layer = Interpolation2d(in_dim, out_dim, 3, pool_kernel_size).cuda()
    y = layer(x)
    
    assert y.shape == (batch_size, output_size, output_size, out_dim)
    print('PASSED:', __file__)
        