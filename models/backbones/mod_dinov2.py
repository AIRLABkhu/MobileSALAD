import torch
import torch.nn as nn
import math

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
        out_features = out_features 
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
            nn.Linear(embed_dim // 4, 1),
        )

    def forward(self, x, policy):
        x = self.in_conv(x)
        B, N, C = x.size()
        local_x = x[:, :, :C//2]
        global_x = (x[:, :, C//2:] * policy).sum(dim=1,
                                                 keepdim=True) / torch.sum(policy, dim=1, keepdim=True)
        x = torch.cat([local_x, global_x.expand(B, N, C//2)], dim=-1)
        return self.out_conv(x)

# class TPS_merge(nn.Module):
#     # from pruned tokens to keep tokens
#     def __init__(self, l2_norm=False, temperature=1) -> None:
#         super().__init__()
#         self.l2_norm = l2_norm
#         self.temperature = temperature

#     def get_sim(x, y, eps=1e-6, mask_eye=-100, l2_norm=True):

#         if y is None:
#             y = x
#         if l2_norm:
#             x = x / (x.norm(dim=-1, keepdim=True) + eps)
#             y = y / (y.norm(dim=-1, keepdim=True) + eps)

#         sim = torch.bmm(x, y.permute(0, 2, 1))
#         if mask_eye is not None:
#             sim.masked_fill_(
#                 torch.eye(x.size(1), device=x.device).unsqueeze(0).bool(), mask_eye)
#         return sim

#     def forward(self, x, y):
#         cos_sim = self.get_sim(y, x, mask_eye=None, l2_norm=False)
#         sim_th = cos_sim.amax(dim=2, keepdims=True)
#         mask = (cos_sim == sim_th).float()

#         # N, pruned token dim, keep token dim
#         cos_sim = mask * cos_sim

#         # N,keep token dim, pruned_token dim
#         mask = mask.permute(0, 2, 1)
#         cos_sim = cos_sim.permute(0, 2, 1)
#         numerator = torch.exp(cos_sim) * mask
#         denominator = math.e + numerator.sum(dim=-1, keepdims=True)
#         x = x * (math.e / denominator) + \
#             torch.bmm(numerator / denominator, y)

#         return x

    



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
            num_trainable_blocks: list=[3, 6, 9],
            norm_layer=False,
            return_token=False,
            masking_rate: float=0.2,
            val_masking_rate=None,
            masking_mode: str=None,
        ):
        super().__init__()

        assert model_name in DINOV2_ARCHS.keys(), f'Unknown model name {model_name}'
        self.model = torch.hub.load('facebookresearch/dinov2', model_name)
        self.num_channels = DINOV2_ARCHS[model_name]
        self.num_trainable_blocks = num_trainable_blocks
        self.norm_layer = norm_layer
        self.return_token = return_token
        self.masking_mode = masking_mode

        self.masking_rate = masking_rate
        self.val_masking_rate = val_masking_rate

        self.img_size = img_size
        self.patch_size = 14
        self.num_patches = (img_size // 14)**2

        self.kept_patch_list = [int(self.num_patches * (1-self.masking_rate)**(i+1)) for i in range(len(self.num_trainable_blocks))]
        self.kept_patches = int((self.kept_patch_list[-1]) ** 0.5)
        self.kept_patch_list[-1] = self.kept_patches**2
        self.num_masks = int(self.num_patches - (self.kept_patches ** 2))
        self.kept_patches_row = int((self.num_patches - self.num_masks) ** 0.5)

        self.selectors = nn.ModuleList([Mlp(in_features=self.num_channels, out_features=1) for _ in self.num_trainable_blocks])

        if self.val_masking_rate == None:
            self.val_masking_rate = self.masking_rate
            
        self.val_kept_patch_list = [int(self.num_patches * (1-self.val_masking_rate)**(i+1)) for i in range(len(self.num_trainable_blocks))]
        self.val_kept_patches = int((self.val_kept_patch_list[-1]) ** 0.5)
        self.val_kept_patch_list[-1] = self.val_kept_patches**2
        self.val_num_masks = int(self.num_patches - (self.val_kept_patches ** 2))
        # self.kept_patches_row = int((self.num_patches - self.num_masks) ** 0.5)

        # Dynamic ViT Precitor
        # self.selectors = nn.ModuleList([PredictorLG(embed_dim=self.num_channels) for _ in self.num_trainable_blocks])





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
        # print(masked_f.shape)
        
        return x, indices


    def TPS_merge(self, x, y):
        # x: keep token
        # y: pruned token

        def get_sim(x, y, eps=1e-6, mask_eye=-100, l2_norm=True):

            if y is None:
                y = x
            if l2_norm:
                x = x / (x.norm(dim=-1, keepdim=True) + eps)
                y = y / (y.norm(dim=-1, keepdim=True) + eps)

            sim = torch.bmm(x, y.permute(0, 2, 1))
            if mask_eye is not None:
                sim.masked_fill_(
                    torch.eye(x.size(1), device=x.device).unsqueeze(0).bool(), mask_eye)
            return sim

        cos_sim = get_sim(y, x, mask_eye=None, l2_norm=False)
        sim_th = cos_sim.amax(dim=2, keepdims=True)
        mask = (cos_sim == sim_th).float()

        # N, pruned token dim, keep token dim
        cos_sim = mask * cos_sim

        # N,keep token dim, pruned_token dim
        mask = mask.permute(0, 2, 1)
        cos_sim = cos_sim.permute(0, 2, 1)
        numerator = torch.exp(cos_sim) * mask
        denominator = math.e + numerator.sum(dim=-1, keepdims=True)
        x = x * (math.e / denominator) + \
            torch.bmm(numerator / denominator, y)

        return x
        
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
        indices = mask_hard.detach().bool()
        indices = indices.expand_as(masked_f)
        masked_f = masked_f[indices].reshape(B, -1, DIM)

        # # pruned token
        # pruned_hard = 1 - mask_hard
        
        # pruned_f = f * pruned_hard
        # pruned_f = pruned_f[~indices].reshape(B, -1, DIM)

        # # merge (TPS module)
        # merged_f = self.TPS_merge(torch.Tensor(masked_f), pruned_f)
        
        # x = torch.cat([t, merged_f], dim=1)
        x = torch.cat([t, masked_f], dim=1)

        return x, indices

    # def merge_and_prune(self, x, i):
    #     t = x[:, 0, None]
    #     f = x[:, 1:]

    #     B, NP, DIM = f.shape

    #     idx = self.num_trainable_blocks.index(i)

    #     selected_prob = self.selectors[idx](f)

    #     # policy = torch.ones(B, NP, 1, dtype=f.dtype, device=f.device)
    #     # selected_prob = self.selector(f, policy)

    #     mask_hard = gumbel_topk(selected_prob, k=self.kept_patch_list[idx], dim=1)

    #     metric = f / f.norm(dim=-1, keepdim=True)

    #     selected_f = metric * mask_hard
    #     pruned_f = metric * (1-mask_hard)

    #     indices = mask_hard.detach().bool()
    #     indices = indices.expand_as(selected_f)

    #     selected_f = selected_f[indices].reshape(B, -1, DIM)
    #     pruned_f = pruned_f[~indices].reshape(B, -1, DIM)

    #     scores = selected_f @ pruned_f.transpose(-1, -2)
    #     node_max, node_idx = scores.max(dim=-1)
    #     edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

    #     r = int(self.kept_patch_list[idx] / 2)

    #     unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
    #     src_idx = edge_idx[..., :r, :]  # Merged Tokens
    #     dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)

    #     unm = selected_f.gather(dim=-2, index=unm_idx.expand(1, 3, 12))
    #     src = selected_f.gather(dim=-2, index=src_idx.expand(1, 3, 12))

    #     # selected_pruned = torch.index_select(pruned_f, dim=1, index=dst_idx.squeeze())

    #     un_dst_idx = torch.unique(dst_idx)
    #     un_mask = torch.ones_like(pruned_f, dtype=torch.bool)
    #     un_mask[un_dst_idx] = 0
    #     # unselected_pruned = torch.masked_select(pruned_f, un_mask)

    #     merged_global_token = torch.mean(torch.cat([t, unselected_pruned], dim = 1), dim=1)

    #     merged_feature  = src + selected_pruned
    #     merged_feature /= 2

    #     selected_f[src_idx] = merged_feature

    #     x = torch.cat([merged_global_token, selected_f], dim=1)

    #     return x, indices


    def forward(self, x):
        B, C, H, W = x.shape
        # total_ = torch.ones_like(x)
        # mask_ = total_.clone()

        x = self.model.prepare_tokens_with_masks(x)

        for i, blk in enumerate(self.model.blocks):
            # First blocks are frozen
            if i < self.num_trainable_blocks[0]:
                with torch.no_grad():
                    x = blk(x)
            elif i in self.num_trainable_blocks:
                if i == self.num_trainable_blocks[0]:
                    x = x.detach()
                    if self.masking_mode is None:
                        pr_x, ind = self.prune(x, i)
                    else:
                        pr_x, ind = self.prune_patch(x, i)
                    total_ = ind.clone()
                    mask_ = total_.clone()

                else:
                    if self.masking_mode is None:
                        pr_x, ind = self.prune(pr_x, i)
                    else:
                        pr_x, ind = self.prune_patch(pr_x, i)
                    total_[mask_] = ind.reshape(-1)
                    mask_ = total_.clone()

                # after pruing, go to transformer block
                pr_x = blk(pr_x)
                x = blk(x)
            else:
                pr_x = blk(pr_x)
                x = blk(x)

        if self.norm_layer:
            x = self.model.norm(x)
            pr_x = self.model.norm(pr_x)
        
        t = x[:, 0]
        f = x[:, 1:]

        pr_t = pr_x[:, 0]
        pr_f = pr_x[:, 1:]

        f = f * total_
        total_ = total_.detach().bool()
        f = f[total_].reshape(B, -1, self.num_channels)

        if self.training:
            pr_f = pr_f.reshape((B, self.kept_patches, self.kept_patches, self.num_channels)).permute(0, 3, 1, 2)
            gt_f = f.reshape((B, self.kept_patches, self.kept_patches, self.num_channels)).permute(0, 3, 1, 2)

        else:
            pr_f = pr_f.reshape((B, self.val_kept_patches, self.val_kept_patches, self.num_channels)).permute(0, 3, 1, 2)
            gt_f = f.reshape((B, self.val_kept_patches, self.val_kept_patches, self.num_channels)).permute(0, 3, 1, 2) 

        if self.return_token:
            return pr_f, pr_t, gt_f, t
        return pr_f, gt_f


if __name__ == '__main__':
    import timm
    DEVICE = 6
    
    sample = torch.randn(2, 3, 224, 224).cuda(DEVICE)
    net1 = mod_DINOv2(return_token=True).cuda(DEVICE)
    
    f, t = net1(sample)
    