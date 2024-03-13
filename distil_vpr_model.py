import os
from typing import Iterator
from pathlib import Path
import yaml

import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.optim import lr_scheduler, optimizer
from torchvision import transforms
from torchvision.transforms import Resize, InterpolationMode

import utils
from models import helper
from models.modules import Interpolation2d


class CBAM(nn.Module):
    def __init__(self, input_patch, input_dim, output_patch, output_dim, last_kernel_size: int=7):
        super(CBAM, self).__init__()

        self.apply_pool = (input_patch != output_patch)

        self.last_kernel_size = last_kernel_size
        self.kernel_size = input_patch - output_patch - last_kernel_size + 2

        if self.apply_pool:
            self.max_pool = nn.MaxPool2d(kernel_size=self.kernel_size, stride=1)
            self.avg_pool = nn.AvgPool2d(kernel_size=self.kernel_size, stride=1)
            self.last_conv = nn.Conv2d(input_dim * 2, output_dim, kernel_size=last_kernel_size)
        else:
            self.max_pool, self.avg_pool = None, None
            self.last_conv = nn.Conv2d(input_dim, output_dim, kernel_size=1)

    def forward(self, x):
        '''
        Inputs:
            batch_size, num_heads, patch_size, patch_size

        Returns:
            batch_size, student_num_heads, student_patch_size, student_patch_size
        '''
        if self.apply_pool:
            x_max = self.max_pool(x)  # ..................| bs, nh, std_ps + 7 - 1, std_ps + 7 - 1
            x_avg = self.avg_pool(x) 
            x = torch.cat([x_max, x_avg], dim=1)  # ......| bs, nh * 2, std_ps + 7 - 1, std_ps + 7 - 1
            return torch.sigmoid(self.last_conv(x))  # ...| bs, std_nh, std_ps, std_ps
        else:
            return torch.sigmoid(self.last_conv(x))  # ...| bs, std_nh, std_ps, std_ps  
        
def mapping_index(idx_list):
    """_summary_

    Args:
        idx_list (list) : 3 torch.Tensor.
            idx_list[0] : first pruning 256 to 203
            idx_list[1] : second pruning 203 to 161
            idx_list[2] : final purning 161 to 121
            
    Outputs:
        re_mapping_mid (torch.Tensor) : mapped index 162 to 256
        re_mapping_final (torch.Tensor) : mapped index 121 to 256
    """
    target_idx = idx_list[0]
    mid_idx = idx_list[1]
    final_idx = idx_list[2]
    
    # Sort target index and cache sorted indices
    sorted_target_idx = torch.sort(target_idx, dim=1)[0]
    
    # Create mapping for mid index
    mid_mapping = sorted_target_idx.gather(dim=1, index=mid_idx)
    
    # Create mapping for final index
    final_mapping = mid_mapping.gather(dim=1, index=final_idx)
    
    return [idx_list[0], mid_mapping, final_mapping]

def get_spatial_feature(mapped_idx_list, prob_list, x_list):
    """_summary_

    Args:
        mapped_idx_list (list): 
            __len__() : 3
        prob_list (list): 
            __len__() : 3
        x_list (_type_): 
            __len__() : 3

    Returns:
        spatial_feature_list (list): 
            __len__(3)
    """
    spatial_feature_list = []
    token_list = []
    for idx, prob, x in zip(mapped_idx_list, prob_list, x_list):
        B, NP= idx.shape
        _, _, D = x.shape
        t = x[:, 0]
        f = x[:, 1:]
        empty_tensor = torch.zeros(B, 256, D).to(device=x.device)
        expanded_idx = idx.unsqueeze(-1).expand(B, NP, D)
        expanded_prob = prob.unsqueeze(-1).expand(B, NP, D)

        empty_tensor.scatter_(1, expanded_idx, expanded_prob * f)
        spatial_feature_list.append(empty_tensor)
        token_list.append(t)
        
    return spatial_feature_list, token_list


def get_spatial_feture_list(keeping_zip):
    idx_list, prob_list, feature_list = zip(*keeping_zip)
    mapped_idx_list = mapping_index(idx_list)
    spatial_feature_list, token_list = get_spatial_feature(mapped_idx_list, prob_list, feature_list)

    return token_list, spatial_feature_list


DINOV2_DIMS = {
    'dinov2_vits14': 384,
    'dinov2_vitb14': 768,
    'dinov2_vitl14': 1024,
    'dinov2_vitg14': 1536,
}

DINOV2_N_HEADS = {
    'dinov2_vits14': 6,
    'dinov2_vitb14': 12,
    'dinov2_vitl14': 16,
    'dinov2_vitg14': 24,
}


class DistillationModel(pl.LightningModule):
    """This is the main model for Visual Place Recognition
    we use Pytorch Lightning for modularity purposes.

    Args:
        pl (_type_): _description_
    """

    def __init__(self,
        #---- Backbone
        teacher_config: str,
        # backbone_arch='dinov2_vitb14',
        # backbone_config={},
        
        student_arch=None,
        student_config={},
        
        # #---- Aggregator
        # agg_arch='SALAD',
        # agg_config={},
        
        #---- Train hyperparameters
        lr=0.03, 
        optimizer='sgd',
        weight_decay=1e-3,
        momentum=0.9,
        lr_sched='linear',
        lr_sched_args = {
            'start_factor': 1,
            'end_factor': 0.2,
            'total_iters': 4000,
        },
        
        #----- Loss
        loss_name='MultiSimilarityLoss', 
        miner_name='MultiSimilarityMiner', 
        miner_margin=0.1,
        faiss_gpu=False,
        faiss_device=0,
        distill_loss_rate=1,
    ):
        super().__init__()
        teacher_config = os.path.join(teacher_config, 'lightning_logs', 'version_0')
        hparams_filename = os.path.join(teacher_config, 'hparams.yaml')
        ckpt_filename = os.path.join(teacher_config, 'checkpoints', 'last.ckpt')
        
        teacher_hparams = yaml.full_load(Path(hparams_filename).read_text())
        teacher_ckpt = torch.load(ckpt_filename, map_location='cpu')
        
        teacher_state_dict = teacher_ckpt['state_dict']
        get_state_dict = lambda x: {key.replace(x, ''): val for key, val in teacher_state_dict.items() if key.startswith(x)}
        backbone_state_dict = get_state_dict('backbone.')
        aggregator_state_dict = get_state_dict('aggregator.')

        # Backbone
        self.encoder_arch = teacher_hparams['backbone_arch']
        self.backbone_config = teacher_hparams['backbone_config']

        self.student_arch = student_arch
        self.student_config = {
            **student_config,
            'model_name': student_arch,
            # 'num_trainable_blocks': self.backbone_config['num_trainable_blocks'][-1:],
            'masking_ratio': 0,
        }
        
        # Aggregator
        self.agg_arch = teacher_hparams['agg_arch']
        self.agg_config = teacher_hparams['agg_config']

        # Train hyperparameters
        self.lr = lr
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.lr_sched = lr_sched
        self.lr_sched_args = lr_sched_args

        # Loss
        self.loss_name = loss_name
        self.miner_name = miner_name
        self.miner_margin = miner_margin
        
        self.save_hyperparameters() # write hyperparams into a file
        
        self.loss_fn = utils.get_loss(loss_name)
        self.miner = utils.get_miner(miner_name, miner_margin)
        self.batch_acc = [] # we will keep track of the % of trivial pairs/triplets at the loss level 
        self.criterion = torch.nn.L1Loss()
        self.faiss_gpu = faiss_gpu
        self.faiss_device = faiss_device
        self.distill_loss_rate = distill_loss_rate
        
        # ----------------------------------
        # get the backbone and the aggregator
        self.resize_fn = transforms.Resize(size=(self.student_config['img_size'],))
        
        self.backbone = helper.get_backbone(self.encoder_arch, self.backbone_config)
        self.student_aggregator = helper.get_aggregator(self.agg_arch, self.agg_config)
        self.teacher_aggregator = helper.get_aggregator(self.agg_arch, self.agg_config)
        self.student = helper.get_backbone(self.student_arch, self.student_config)
        
        backbone_num_heads = DINOV2_N_HEADS[self.encoder_arch]
        student_num_heads = DINOV2_N_HEADS[self.student_arch]
        # self.attn_align_fn = nn.ModuleList(CBAM(input_patch=p+1, input_dim=DINOV2_N_HEADS[self.encoder_arch], \
        #     output_patch=self.backbone.keep_patch_list[-1]+1, output_dim=DINOV2_N_HEADS[self.student_arch]) for p in self.backbone.keep_patch_list)
        
        # self.feature_align_fn = nn.ModuleList(nn.Conv2d(DINOV2_DIMS[self.encoder_arch], DINOV2_DIMS[self.student_arch], kernel_size=(1,1)) for _ in self.backbone.keep_patch_list)
        # self.feature_interpolate_fn = nn.ModuleList(Interpolation2d(in_dim=768, out_dim=576, conv_kernel_size=3, pool_kernel_size=6, nonlinearity=nn.GELU)for _ in self.backbone.keep_patch_list)
        self.feature_interpolate_fn = Interpolation2d(in_dim=768, out_dim=576, conv_kernel_size=3, pool_kernel_size=6, nonlinearity=nn.GELU)
        self.feature_align_fn = nn.ModuleList(nn.Conv2d(DINOV2_DIMS[self.student_arch], 576, kernel_size=(1,1)) for _ in self.backbone.keep_patch_list)
        
        self.student.align_fn = nn.Linear(DINOV2_DIMS[self.student_arch], DINOV2_DIMS[self.encoder_arch])
        
        self.backbone.load_state_dict(backbone_state_dict)
        self.student_aggregator.load_state_dict(aggregator_state_dict)
        self.teacher_aggregator.load_state_dict(aggregator_state_dict)
        # For validation in Lightning v2.0.0
        self.val_outputs = []
        
    # the forward pass of the lightning model
    def forward(self, x):
        if self.training:
            with torch.no_grad():
                t_t, t_f, out_pred_prob, keep_zip = self.backbone.eval()(x, role='teacher', mode='distill')
            distill_t_token, distill_t_feature = get_spatial_feture_list(keep_zip)
            interpolated_t_feature = [self.feature_interpolate_fn[i](feature.reshape(-1, 16, 16, 768))for i, feature in enumerate(distill_t_feature)]
            interpolated_t_feature = torch.cat(interpolated_t_feature, dim=0)
            s_t, s_f, s_f_zip = self.student(self.resize_fn(x), role='student', mode='distill')
            distill_s_feature = torch.cat([self.feature_align_fn[i](s_[:, 1:].permute(0,2,1).reshape(-1, 384, 11, 11)) for i, s_ in enumerate(s_f_zip)], dim=0)
            return self.student_aggregator((s_f, s_t)), self.teacher_aggregator((t_f, t_t)), distill_s_feature, interpolated_t_feature
        else:
            p_t, p_f, _ = self.student(self.resize_fn(x), role='student', mode='distill')
            return self.student_aggregator((p_f, p_t))

    @utils.yield_as(list)
    def parameters(self, recurse: bool=True) -> Iterator[Parameter]:
        yield from self.student.parameters(recurse=recurse)
        yield from self.feature_align_fn.parameters(recurse=recurse)
        yield from self.feature_interpolate_fn.parameters(recurse=recurse)
        yield from self.student_aggregator.parameters(recurse=recurse)
    
    # configure the optimizer 
    def configure_optimizers(self):
        if self.optimizer.lower() == 'sgd':
            optimizer = torch.optim.SGD(
                self.parameters(), 
                lr=self.lr, 
                weight_decay=self.weight_decay, 
                momentum=self.momentum
            )
        elif self.optimizer.lower() == 'adamw':
            optimizer = torch.optim.AdamW(
                self.parameters(), 
                lr=self.lr, 
                weight_decay=self.weight_decay
            )
        elif self.optimizer.lower() == 'adam':
            optimizer = torch.optim.Adam(
                self.parameters(), 
                lr=self.lr, 
                weight_decay=self.weight_decay
            )
        else:
            raise ValueError(f'Optimizer {self.optimizer} has not been added to "configure_optimizers()"')
        
        if self.lr_sched.lower() == 'multistep':
            scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_sched_args['milestones'], gamma=self.lr_sched_args['gamma'])
        elif self.lr_sched.lower() == 'cosine':
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, self.lr_sched_args['T_max'])
        elif self.lr_sched.lower() == 'linear':
            scheduler = lr_scheduler.LinearLR(
                optimizer,
                start_factor=self.lr_sched_args['start_factor'],
                end_factor=self.lr_sched_args['end_factor'],
                total_iters=self.lr_sched_args['total_iters']
            )

        return [optimizer], [scheduler]
    
    # configure the optizer step, takes into account the warmup stage
    def optimizer_step(self,  epoch, batch_idx, optimizer, optimizer_closure):
        # warm up lr
        optimizer.step(closure=optimizer_closure)
        self.lr_schedulers().step()
        
    #  The loss function call (this method will be called at each training iteration)
    def loss_function(self, s_desc, t_desc, labels, distill_s, distill_t, log_accuracy: bool=False):
        ## This part is same as pruning part.
        # we mine the pairs/triplets if there is an online mining strategy
        if self.miner is not None:
            miner_outputs = self.miner(s_desc, labels)
            loss = self.loss_fn(s_desc, labels, miner_outputs)
            
            # calculate the % of trivial pairs/triplets 
            # which do not contribute in the loss value
            nb_samples = s_desc.shape[0]
            nb_mined = len(set(miner_outputs[0].detach().cpu().numpy()))
            batch_acc = 1.0 - (nb_mined/nb_samples)
        else: # no online mining
            loss = self.loss_fn(s_desc, labels)
            batch_acc = 0.0
            if type(loss) == tuple: 
                # somes losses do the online mining inside (they don't need a miner object), 
                # so they return the loss and the batch accuracy
                # for example, if you are developing a new loss function, you might be better
                # doing the online mining strategy inside the forward function of the loss class, 
                # and return a tuple containing the loss value and the batch_accuracy (the % of valid pairs or triplets)
                loss, batch_acc = loss

        if log_accuracy:
            # keep accuracy of every batch and later reset it at epoch start
            self.batch_acc.append(batch_acc)
            # log it
            self.log('b_acc', sum(self.batch_acc) /
                    len(self.batch_acc), prog_bar=True, logger=True)
            
        # dist_desc_loss = (1 - F.cosine_similarity(s_desc, t_desc, dim=-1).mean())
        distill_loss = nn.functional.mse_loss(distill_s, distill_t)
        distill_loss += nn.functional.mse_loss(s_desc, t_desc)

        loss = loss + self.distill_loss_rate * distill_loss
        return loss, distill_loss
    
    # This is the training step that's executed at each iteration
    def training_step(self, batch, batch_idx):
        places, labels = batch
        
        # Note that GSVCities yields places (each containing N images)
        # which means the dataloader will return a batch containing BS places
        BS, N, ch, h, w = places.shape
        # assert N == 2  # Our method forces each place to have exactly two images in a mini-batch. 
        
        # reshape places and labels
        # data 를 다시...
        images = places.view(BS*N, ch, h, w)
        labels = labels.view(-1)

        # Feed forward the ba6tch to the model
        # Here we are calling the method forward that we defined above

        s_desc, t_desc, distill_s, distill_t = self(images)

        if torch.isnan(s_desc).any():
            raise ValueError('NaNs in descriptors (student descriptor)')
        if torch.isnan(t_desc).any():
            raise ValueError('NaNs in descriptors (teacher descriptor)')

        # Call the loss_function we defined above
        loss, distill_loss = self.loss_function(s_desc, t_desc, labels, distill_s, distill_t, log_accuracy=True) 
        self.log('loss', loss.item(), logger=True, prog_bar=True)
        self.log('distill_loss', distill_loss.item(), logger=False, prog_bar=True)
        return {'loss': loss}
    
    def on_train_epoch_end(self):
        # we empty the batch_acc list for next epoch
        self.batch_acc = []

    # For validation, we will also iterate step by step over the validation set
    # this is the way Pytorch Lghtning is made. All about modularity, folks.
    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        places, _ = batch
        descriptors = self(places)

        if torch.isnan(descriptors).any():
            raise ValueError('NaNs in descriptors')
        self.val_outputs[dataloader_idx].append(descriptors.detach().cpu().to(dtype=torch.float32))
        return descriptors.detach().cpu()
    
    def on_validation_epoch_start(self):
        # reset the outputs list
        self.val_outputs = [[] for _ in range(len(self.trainer.datamodule.val_datasets))]
    
    def on_validation_epoch_end(self):
        """this return descriptors in their order
        depending on how the validation dataset is implemented 
        for this project (MSLS val, Pittburg val), it is always references then queries
        [R1, R2, ..., Rn, Q1, Q2, ...]
        """
        val_step_outputs = self.val_outputs

        dm = self.trainer.datamodule
        # The following line is a hack: if we have only one validation set, then
        # we need to put the outputs in a list (Pytorch Lightning does not do it presently)
        if len(dm.val_datasets)==1: # we need to put the outputs in a list
            val_step_outputs = [val_step_outputs]
        
        for i, (val_set_name, val_dataset) in enumerate(zip(dm.val_set_names, dm.val_datasets)):
            feats = torch.concat(val_step_outputs[i], dim=0)
            
            if 'pitts' in val_set_name:
                # split to ref and queries
                num_references = val_dataset.dbStruct.numDb
                positives = val_dataset.getPositives()
            elif 'msls' in val_set_name:
                # split to ref and queries
                num_references = val_dataset.num_references
                positives = val_dataset.pIdx
            elif 'nordland' in val_set_name:
                # split to ref and queries
                num_references = val_dataset.num_references
                positives = val_dataset.ground_truth
            elif 'sped' in val_set_name:
                # split to ref and queries
                num_references = val_dataset.num_references
                positives = val_dataset.ground_truth
            else:
                print(f'Please implement validation_epoch_end for {val_set_name}')
                raise NotImplemented

            r_list = feats[ : num_references]
            q_list = feats[num_references : ]

            pitts_dict = utils.get_validation_recalls(
                r_list=r_list, 
                q_list=q_list,
                k_values=[1, 5, 10, 15, 20, 50, 100],
                gt=positives,
                print_results=True,
                dataset_name=val_set_name,
                faiss_gpu=self.faiss_gpu,
                faiss_device = self.faiss_device
            )
            del r_list, q_list, feats, num_references, positives

            self.log(f'{val_set_name}/R1', pitts_dict[1], prog_bar=False, logger=True, sync_dist=True)
            self.log(f'{val_set_name}/R5', pitts_dict[5], prog_bar=False, logger=True, sync_dist=True)
            self.log(f'{val_set_name}/R10', pitts_dict[10], prog_bar=False, logger=True, sync_dist=True)
        print('\n\n')

        # reset the outputs list
        self.val_outputs = []
