import torch
import torch.nn as nn

from utils import gumbel_topk

DINOV2_ARCHS = {
    'dinov2_vits14': 384,
    'dinov2_vitb14': 768,
    'dinov2_vitl14': 1024,
    'dinov2_vitg14': 1536,
}

class Mlp(nn.Module):
    def __init__(self, in_features, out_features=None, act_layer=nn.GELU, drop=0., group=1):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = in_features 

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        # B, NP, DIM = x.shape
        # x = x.reshape(B, NP * DIM)

        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        x = x.squeeze()
        return x

class Predictor(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.in_conv = nn.Sequential(
            # nn.LayerNorm(self.embed_dim * self.num_patch * 2),
            nn.Linear(self.in_features , self.out_features),
            # nn.Tanh()
        )

    def forward(self, x):
        # x1, x2 ---> B, NP, DIM
        B, NP, DIM = x.shape
        # x = x.reshape(B, NP * DIM)
        x = self.in_conv(x) # x ---> B* NP
        # softmax를 한 번 거쳐서 out을 해야할지...?
        x = x.squeeze()
        return x

class mod_DINOv2(nn.Module):
    """
    DINOv2 model

    Args:
        model_name (str): The name of the model architecture 
            should be one of ('dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14')
        num_trainable_blocks (int): The number of last blocks in the model that are trainable.
        norm_layer (bool): If True, a normalization layer is applied in the forward pass.
        return_token (bool): If True, the forward pass returns both the feature map and the token.
    """
    def __init__(
            self,
            model_name='dinov2_vitb14',
            img_size: int=224,
            num_trainable_blocks=4,
            norm_layer=False,
            return_token=False,
            masking_rate: float=0.4
        ):
        super().__init__()

        assert model_name in DINOV2_ARCHS.keys(), f'Unknown model name {model_name}'
        self.model = torch.hub.load('facebookresearch/dinov2', model_name)
        self.num_channels = DINOV2_ARCHS[model_name]
        self.num_trainable_blocks = num_trainable_blocks
        self.norm_layer = norm_layer
        self.return_token = return_token

        self.img_size = img_size
        self.patch_size = 14
        self.num_patches = (img_size // 14)**2

        self.kept_patches = int((self.num_patches - int(self.num_patches * masking_rate)) ** 0.5)
        self.num_masks = int(self.num_patches - (self.kept_patches ** 2))
        self.masking_rate = self.num_masks / self.num_patches
        self.kept_patches_row = int((self.num_patches - self.num_masks) ** 0.5)

        self.predictor = Predictor(in_features=self.num_channels, out_features=1)
        self.selector = Predictor(in_features=self.num_channels, out_features=1)
        # self.predictor = Mlp(in_features=self.num_channels , out_features=1)


    def calc_cosine(
        self, 
        f1: torch.Tensor, 
        f2: torch.Tensor, 
    ) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        '''
        x1, x2: B, NP+1, DIM
        only training.
        '''
        if f1.shape != f2.shape:
            raise RuntimeError('The shape of the two tensors must equal.')
        
        simm = torch.nn.functional.cosine_similarity(f1, f2, dim=2)  #...| B/2, NP
        simm = torch.cat([simm, simm], dim=0) #...| B, NP
        
        return simm.unsqueeze(-1) #...| B, NP, 1
    
    def predict_cosine(
        self, 
        f: torch.Tensor,
    ):
        # Predictor
        pred_simm = self.predictor.forward(f)  #...............................| B, NP
        pred_simm = pred_simm.unsqueeze(-1)  #.................................| B, NP, 1
        
        return pred_simm

    def prune_patch(
        self, 
        f: torch.Tensor, #.....................................................| B, NP, DIM
    ):
        B, NP, DIM = f.shape
        NK = NP - self.num_masks

        cosine_sim = - torch.nn.functional.cosine_similarity(f.unsqueeze(1), f.unsqueeze(2), dim=3)
        mean_sim = torch.mean(cosine_sim, dim=2) # BS, NP

        mask_hard = gumbel_topk(mean_sim, k=NK, dim=1)
        mask_hard = mask_hard.unsqueeze(-1)
        masked_f = f * mask_hard
        
        indices = mask_hard.detach().bool()  # ................| B, NP, 1
        indices = indices.expand_as(masked_f)  # .............| B, NP, DIM
        masked_f = masked_f[indices].reshape(B, NK, DIM)
        
        return masked_f, indices
    
    def forward(self, x):
        B, C, H, W = x.shape

        x = self.model.prepare_tokens_with_masks(x)
        # First blocks are frozen
        # forward_pre
        with torch.no_grad():
            for blk in self.model.blocks[:-self.num_trainable_blocks]:
                x = blk(x)
        x = x.detach()

        t = x[:, 0, None]
        f = x[:, 1:] # BS, NP, DIM

        pruned_f, indices = self.prune_patch(f)
        pruned_x = torch.cat([t, pruned_f], dim=1)

        # Last blocks are trained
        for blk in self.model.blocks[1-self.num_trainable_blocks:]:
            pruned_x = blk(pruned_x)
        if self.norm_layer:
            pruned_x = self.model.norm(pruned_x)
        
        pruned_t = pruned_x[:, 0]
        pruned_f = pruned_x[:, 1:]
        pruned_f = pruned_f.reshape((B, self.kept_patches, self.kept_patches, self.num_channels)).permute(0, 3, 1, 2) 
        
        outputs = [pruned_f]
        if self.return_token:
            outputs.append(pruned_t)
        
        if self.training:
            with torch.no_grad():
                gt_x = x
                for blk in self.model.blocks[1-self.num_trainable_blocks:]:
                    gt_x = blk(gt_x)
                if self.norm_layer:
                    gt_x = self.model.norm(gt_x)
            
            gt_t = gt_x[:, 0]
            gt_f = gt_x[:, 1:] # BS, NP, DIM
            gt_f = gt_f[indices].reshape_as(pruned_f)
            outputs.append(gt_f)
            if self.return_token:
                outputs.append(gt_t)

        return tuple(outputs)  # pruned_f, pruned_t, gt_f, gt_t


if __name__ == '__main__':
    import timm
    DEVICE = 6
    
    sample = torch.randn(2, 3, 224, 224).cuda(DEVICE)
    net1 = mod_DINOv2(return_token=True).cuda(DEVICE)
    
    f, t = net1(sample)
    