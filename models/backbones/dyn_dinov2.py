import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from utils import gumbel_topk
# from utils import gumbel_softmax
from utils import batch_index_select

DINOV2_ARCHS = {
    'dinov2_vits14': 384,
    'dinov2_vitb14': 768,
    'dinov2_vitl14': 1024,
    'dinov2_vitg14': 1536,
}

    
    
class PredictorLG(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, embed_dim=384):
        super().__init__()
        self.in_conv = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU()
        )

        self.out_conv = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, 2),

            # NOTE: the head should return the logits for gumbel softmax
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, x, policy):
        x = self.in_conv(x)
        B, N, C = x.size()
        local_x = x[:, :, :C//2]
        global_x = (x[:, :, C//2:] * policy).sum(dim=1, keepdim=True) / torch.sum(policy, dim=1, keepdim=True)
        x = torch.cat([local_x, global_x.expand(B, N, C//2)], dim=-1)
        return self.out_conv(x)


class Dyn_DINOv2(nn.Module):
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
            num_trainable_blocks: list=[3, 6, 9],
            norm_layer=False,
            return_token=False,
            keep_ratio: list=[0.75, 0.5, 0.25],
        ):
        super().__init__()

        assert model_name in DINOV2_ARCHS.keys(), f'Unknown model name {model_name}'
        # self.model = torch.hub.load('facebookresearch/dinov2', model_name)  # without policy
        self.model = torch.hub.load('dinov2', model_name, source='local')  # with and without policy
        self.num_channels = DINOV2_ARCHS[model_name]
        self.num_trainable_blocks = num_trainable_blocks
        self.norm_layer = norm_layer
        self.return_token = return_token

        self.img_size = img_size
        self.patch_size = 14
        self.num_patches = (img_size // 14)**2
        self.patch_row = (img_size // 14)

        self.keep_ratio = keep_ratio
        self.keep_patch_list = [int(self.num_patches * i) for i in self.keep_ratio]
        self.keep_patches = int((self.keep_patch_list[-1]) ** 0.5)
        self.keep_patch_list[-1] = self.keep_patches**2

        # Dynamic ViT Precitor
        self.selectors = nn.ModuleList([PredictorLG(embed_dim=self.num_channels) for _ in self.num_trainable_blocks])


    def random_mask(self, B, NP, NK):
        all_tensors_ = []
        for _ in range(B):
            perm = torch.randperm(NP)
            idx = perm[:NK]
            tensor_ = torch.zeros(NP, dtype=torch.float)
            tensor_[idx] = 1
            all_tensors_.append(tensor_)
        
        all_tensors_ = torch.stack(all_tensors_)

        return all_tensors_
    
    def checker_mask(self, B, NP):
        rows = int(NP ** 0.5)

        grid_mask = torch.zeros((rows, rows), dtype=torch.float)
        grid_mask[::2, ::2] = 1
        grid_mask[1::2, 1::2] = 1

        grid_mask = grid_mask.view(-1)
        grid_mask = grid_mask.expand(B, NP)

        return grid_mask

    def prune_patch(
        self, 
        x: torch.Tensor, #.....................................................| B, NP, DIM
        i
    ):
        t = x[:, 0, None]
        f = x[:, 1:]

        B, NP, DIM = f.shape

        idx = self.num_trainable_blocks.index(i)

        # for random mask
        if self.masking_mode == 'random':
            if self.training:
                mask_hard = self.random_mask(B, NP, self.kept_patch_list[idx])
            else:
                mask_hard = self.random_mask(B, NP, self.val_kept_patch_list[idx])
        # for checker mask
        elif self.masking_mode == 'checker':
            mask_hard = self.checker_mask(B, NP)

        mask_hard = mask_hard.unsqueeze(-1)
        mask_hard = mask_hard.to(f.device)

        masked_f = f * mask_hard
        
        indices = mask_hard.detach().bool()  # ................| B, NP, 1
        indices = indices.expand_as(masked_f)  # .............| B, NP, DIM
        masked_f = masked_f[indices].reshape(B, -1, DIM)

        x = torch.cat([t, masked_f], dim=1)
        return x, indices
        
    def prune(self, x, i):
        t = x[:, 0, None]
        f = x[:, 1:]

        B, NP, DIM = f.shape
        # policy = torch.ones(B, NP, 1, dtype=f.dtype, device=f.device)
        idx = self.num_trainable_blocks.index(i)

        selected_prob = self.selectors[idx](f)
        # for dynamic predictor
        # selected_prob = self.selectors[idx](f, policy)
        if len(selected_prob.shape) == 1:
            selected_prob = selected_prob.unsqueeze(0)

        if self.training:
            mask_hard = gumbel_topk(selected_prob, k=self.kept_patch_list[idx], dim=1)
        else:
            mask_hard = gumbel_topk(selected_prob, k=self.val_kept_patch_list[idx], dim=1)

        if len(mask_hard.size()) == 2:
            mask_hard = mask_hard.unsqueeze(-1)
        mask_hard = mask_hard.expand_as(f)
        masked_f = f * mask_hard
        masked_f = masked_f[mask_hard.detach().bool()].reshape(B, -1, DIM)

        x = torch.cat([t, masked_f], dim=1)

        return x, mask_hard

    def forward(self, x):

        x = self.model.prepare_tokens_with_masks(x)

        B, NP, DIM = x.shape # NP means number of patch include token ( 256 + 1 )
        
        out_pred_prob = []
        prev_decision = torch.ones(B, NP-1, 1, dtype=x.dtype, device=x.device)
        policy = torch.ones(B, NP, 1, dtype=x.dtype, device=x.device)
        for i, blk in enumerate(self.model.blocks):
            # early blocks are frozen
            if i < self.num_trainable_blocks[0]:
                with torch.no_grad():
                    if self.training:
                        x = blk(x, policy)
                    else:
                        x = blk(x)
            # pruning 
            elif i in self.num_trainable_blocks:
                spatial_x = x[:, 1:] # except global token
                idx = self.num_trainable_blocks.index(i)
                pred_score = self.selectors[idx](spatial_x, prev_decision).reshape(B, -1, 2) # we will use gumble_topk so we need just 1 dimension
                
                if self.training:
                    hard_keep_decision = F.gumbel_softmax(pred_score, hard=True)[:, :, 0:1] * prev_decision
                    out_pred_prob.append(hard_keep_decision.reshape(B, NP-1))
                    token_policy = torch.ones(B, 1, 1, dtype=hard_keep_decision.dtype, device=hard_keep_decision.device)
                    policy = torch.cat([token_policy, hard_keep_decision], dim=1)
                    x = blk(x, policy=policy)
                    prev_decision = hard_keep_decision
                
                else:
                    score = pred_score[:,:,0]
                    num_keep_node = self.keep_patch_list[idx]
                    keep_policy = torch.argsort(score, dim=1, descending=True)[:, :num_keep_node]
                    token_policy = torch.zeros(B, 1, dtype=keep_policy.dtype, device=keep_policy.device)
                    now_policy = torch.cat([token_policy, keep_policy + 1], dim=1)
                    x = batch_index_select(x, now_policy)
                    prev_decision = batch_index_select(prev_decision, keep_policy)
                    x = blk(x)

        if self.norm_layer:
            x = self.model.norm(x)
            
        t = x[:, 0]
        f = x[:, 1:]
        

        if self.training:
            f = f.reshape((B, self.patch_row, self.patch_row, self.num_channels)).permute(0, 3, 1, 2)
            return t, f, prev_decision.detach(), out_pred_prob

        else:
            f = f.reshape((B, self.keep_patches, self.keep_patches, self.num_channels)).permute(0, 3, 1, 2)
            return t, f



# if __name__ == '__main__':
#     import timm
#     DEVICE = 6
    
#     sample = torch.randn(2, 3, 224, 224).cuda(DEVICE)
#     net1 = mod_DINOv2(return_token=True).cuda(DEVICE)
    
#     f, t = net1(sample)
    