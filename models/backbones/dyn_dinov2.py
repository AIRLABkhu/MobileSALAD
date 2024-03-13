from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

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
            num_trainable_blocks: list=[8,9,10],
            norm_layer=False,
            return_token=False,
            masking_ratio: float= 0.2,
            num_register_tokens: int=0,
        ):
        super().__init__()

        assert model_name in DINOV2_ARCHS.keys(), f'Unknown model name {model_name}'
        # self.model = torch.hub.load('facebookresearch/dinov2', model_name)  # without policy
        self.model = torch.hub.load('dinov2', model_name, source='local')  # with and without policy
        self.model.num_register_tokens = num_register_tokens
        self.num_channels = DINOV2_ARCHS[model_name]
        self.num_trainable_blocks = num_trainable_blocks
        self.norm_layer = norm_layer
        self.return_token = return_token

        self.img_size = img_size
        self.patch_size = 14
        self.num_patches = (img_size // 14)**2
        self.patch_row = (img_size // 14)

        self.masking_ratio = masking_ratio

        self.keep_patch_list = [int(self.num_patches * (1-self.masking_ratio)**(i+1)) for i in range(len(self.num_trainable_blocks))]
        self.keep_patches = int((self.keep_patch_list[-1]) ** 0.5)
        self.keep_patch_list[-1] = self.keep_patches**2

        self.ratio_list = [(1-self.masking_ratio)**(i+1) for i in range(len(self.num_trainable_blocks))]
        self.ratio_list[-1] = self.keep_patch_list[-1] / self.num_patches

        # Dynamic ViT Precitor
        self.selectors = nn.ModuleList([PredictorLG(embed_dim=self.num_channels) for _ in self.num_trainable_blocks])
        self.align_fn = nn.Identity()
        
        if self.model.num_register_tokens > 0:
            self.prepare_registers(self.model.num_register_tokens)
        
    def prepare_registers(self, num_registers: int, init: bool=True):
        assert num_registers > 0
        reg_tokens = self.model.cls_token.data.clone()
        reg_tokens = reg_tokens.repeat(1, num_registers, 1)
        
        if init:
            nn.init.normal_(reg_tokens, std=1e-6)
        
        self.model.num_register_tokens = num_registers
        self.model.register_tokens = nn.Parameter(reg_tokens)

    def forward(self, x, role: Literal['student, teacher']='teacher', mode: Literal['pruning', 'distill']='pruning'):
        x = self.model.prepare_tokens_with_masks(x) # after this, x contain register tokens --> ([cls], [reg], [feature])
        B, NP, DIM = x.shape # NP means number of patch include token ( 256 + 1 ) 
        # maybe if you give the parameter "num_register_tokens", the NP would be (1 + num_reigster_tokens + feature_tokens)
        
        out_pred_prob = []
        keep_zip = []
        prev_decision = torch.ones(B, NP-(1 + self.model.num_register_tokens), 1, dtype=x.dtype, device=x.device)
        
        policy = torch.ones(B, NP, 1, dtype=x.dtype, device=x.device)
        for i, blk in enumerate(self.model.blocks):
            if role == 'teacher':
            # early blocks are frozen
                if i < self.num_trainable_blocks[0]:
                    if self.training:
                        x = blk(x, policy)
                    else:
                        with torch.no_grad():
                            x = blk(x)
                # pruning 
                elif i in self.num_trainable_blocks:
                    spatial_x = x[:, 1 + self.model.num_register_tokens:] # except global token
                    idx = self.num_trainable_blocks.index(i)
                    pred_score = self.selectors[idx](spatial_x, prev_decision).reshape(B, -1, 2) # we will use gumble_topk so we need just 1 dimension

                    if self.training:
                        hard_keep_decision = F.gumbel_softmax(pred_score, hard=True)[:, :, 0:1] * prev_decision
                        out_pred_prob.append(hard_keep_decision.reshape(B, NP-(1+self.model.num_register_tokens)))
                        token_policy = torch.ones(B, 1 + self.model.num_register_tokens, 1, dtype=hard_keep_decision.dtype, device=hard_keep_decision.device)  # TODO: complete
                        policy = torch.cat([token_policy, hard_keep_decision], dim=1)
                        x = blk(x, policy=policy, return_attention=False)
                        prev_decision = hard_keep_decision
                    
                    # Validation Step
                    else:
                        if mode == 'pruning':
                            score = pred_score[:,:,0]
                            num_keep_node = self.keep_patch_list[idx]
                            keep_policy = torch.argsort(score, dim=1, descending=True)[:, :num_keep_node]
                            # token_policy = torch.zeros(B, 1, dtype=keep_policy.dtype, device=keep_policy.device)  # TODO: complete
                            token_policy = torch.arange(1 + self.model.num_register_tokens, dtype=keep_policy.dtype, device=keep_policy.device).repeat(B, 1)
                            now_policy = torch.cat([token_policy, keep_policy + (1 + self.model.num_register_tokens)], dim=1)  # TODO: complete
                            x = batch_index_select(x, now_policy)
                            prev_decision = batch_index_select(prev_decision, keep_policy)
                            x = blk(x, return_attention=False)
                            
                        elif mode == 'distill':
                            score = pred_score[:,:,0]
                            num_keep_node = self.keep_patch_list[idx]
                            
                            keeping_patch = torch.zeros_like(score, dtype=x.dtype, device=x.device)
                            keep_prob, keep_idx  = torch.sort(score, dim=1, descending=True)
                            keep_prob = keep_prob[:,:num_keep_node]
                            keep_idx = keep_idx[:,:num_keep_node]
                            
                            keeping_patch.scatter_(1, keep_idx, 1)
                            out_pred_prob.append(keeping_patch)
                            
                            keep_policy = torch.argsort(score, dim=1, descending=True)[:, :num_keep_node]
                            token_policy = torch.arange(1 + self.model.num_register_tokens, dtype=keep_policy.dtype, device=keep_policy.device).repeat(B, 1)  # TODO: complete
                            now_policy = torch.cat([token_policy, keep_policy + (1 + self.model.num_register_tokens)], dim=1)  # TODO:
                            x = batch_index_select(x, now_policy)
                            prev_decision = batch_index_select(prev_decision, keep_policy)
                            x = blk(x, return_attention=False)
                            
                            keep_zip.append((keep_idx, keep_prob, x))         
                else:
                    if self.training:
                        x = blk(x, policy=policy, return_attention=False)   
                    else:
                        x = blk(x)
                        
                            
            elif role=='student':
                x = blk(x, return_attention=False)
                if i in self.num_trainable_blocks:
                    keep_zip.append(x)
                
        if self.norm_layer:
            x = self.model.norm(x)
        x = self.align_fn(x)
            
        t = x[:, 0]
        f = x[:, 1 + self.model.num_register_tokens:]
        
        if self.training:
            patches_row = int(f.size(1) ** 0.5)
            f = f.reshape((B, patches_row, patches_row, -1)).permute(0, 3, 1, 2)
        else:
            f = f.reshape((B, self.keep_patches, self.keep_patches, -1)).permute(0, 3, 1, 2)
        
        if role == 'teacher':
            if mode == 'distill':
                return t, f, out_pred_prob, keep_zip
            elif mode == 'pruning':
                return t, f, prev_decision.detach(), out_pred_prob
        elif role == 'student':
            return t, f, keep_zip
            
        else:
            raise NameError
