from typing import Any

import torch
try:
    import spconv.pytorch as spconv
except ModuleNotFoundError as e:
    print('ModuleNotFoundError:',
          'Install sparse convolution at', 
          'https://github.com/traveller59/spconv?tab=readme-ov-file#install.')
    exit(0)


class SpConvAdapter(spconv.SubMConv2d):
    def __init__(
        self,
        in_channels: Any,
        out_channels: Any,
        kernel_size: Any,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        indice_key: Any | None = None,
        algo: spconv.ConvAlgo | None = None,
        fp32_accum: bool | None = None,
        large_kernel_fast_algo: bool = False,
        name: Any | None = None):
        '''
        INPUTS:
            `x`: a dense PyTorch tensor with shape of `(N, H, W, C)`.
            `squeeze_zeros`: indicates to squeeze zero tokens (default: `False`).
            
        RETURNS: 
            `y`: a dense PyTorch tensor with shape of 
                (N, H, W, C) if `squeeze_zeros=False` or
                (N, D, C) if `squeeze_zeros=True`.
        '''
        super(SpConvAdapter, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            indice_key=indice_key,
            algo=algo,
            fp32_accum=fp32_accum,
            large_kernel_fast_algo=large_kernel_fast_algo,
            name=name
        )
        import os
        if 'SPCONV_DEBUG_SAVE_PATH' not in os.environ:
            print('Setup envvar: SPCONV_DEBUG_SAVE_PATH')
            if not os.path.exists('./debug'):
                os.mkdir('./debug')
            os.environ['SPCONV_DEBUG_SAVE_PATH'] = './debug'
        
    def forward(self, x: torch.Tensor, squeeze_zeros: bool=False):
        x = spconv.SparseConvTensor.from_dense(x.to(self.weight.device))
        y = super(spconv.SubMConv2d, self).forward(x)
        y = y.dense(channels_first=False)
        
        if squeeze_zeros:
            with torch.no_grad():
                mask = y.abs().sum(dim=-1).ne(0)
            y = y[mask].reshape(y.size(0), -1, self.out_channels)
        return y
    
    

if __name__ == '__main__':    
    device = 7
    t = torch.randn(64, 16, 16, 768).cuda(device).requires_grad_(True)
    net = SpConvAdapter(768, 384, kernel_size=3).cuda(device)
    out = net(t, squeeze_zeros=False)
    print(f'{out.shape=}')
    out.square().sum().backward()
    print('PASSED:', __file__)
    