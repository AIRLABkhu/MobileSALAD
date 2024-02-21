from typing import Iterator
import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn.parameter import Parameter
from torch.optim import lr_scheduler, optimizer

import utils
from models import helper


class VPRModel(pl.LightningModule):
    """This is the main model for Visual Place Recognition
    we use Pytorch Lightning for modularity purposes.

    Args:
        pl (_type_): _description_
    """

    def __init__(self,
        #---- Backbone
        backbone_arch='dinov2_vitb14',
        backbone_config={},
        
        teacher_arch=None,
        teacher_config={},
        
        #---- Aggregator
        agg_arch='SALAD',
        agg_config={},
        
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
        faiss_device=0
    ):
        super().__init__()

        # Backbone
        self.encoder_arch = backbone_arch
        self.backbone_config = backbone_config

        self.teacher_arch = teacher_arch
        self.teacher_config = teacher_config
        
        # Aggregator
        self.agg_arch = agg_arch
        self.agg_config = agg_config

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
        
        # ----------------------------------
        # get the backbone and the aggregator
        self.backbone = helper.get_backbone(backbone_arch, backbone_config)
        self.aggregator = helper.get_aggregator(agg_arch, agg_config)
        if self.teacher_arch is not None:
            self.teacher = helper.get_teacher(teacher_arch, teacher_config)

        # For validation in Lightning v2.0.0
        self.val_outputs = []
        
    # the forward pass of the lightning model
    def forward(self, x):
        if self.training:
            s_t, s_f, s_prev_decision, s_out_pred_prob = self.backbone(x)
            # s_out = self.aggregator((s_f, s_t))
            with torch.no_grad():
                t_t, t_f = self.teacher(x)
            return self.aggregator((s_f, s_t)), self.aggregator((t_f, t_t)), s_t, s_f, s_prev_decision, s_out_pred_prob
        else:
            p_t, p_f = self.backbone(x)
            return self.aggregator((p_f, p_t))

    @utils.yield_as(list)
    def parameters(self, recurse: bool=True) -> Iterator[Parameter]:
        # yield self.backbone.model.pos_embed
        yield from self.backbone.model.blocks[self.backbone.num_trainable_blocks[0]:].parameters(recurse=recurse)
        yield from self.backbone.model.norm.parameters(recurse=recurse)
        # yield from self.backbone.model.fc_norm.parameters(recurse=recurse)
        # yield from self.backbone.model.head_drop.parameters(recurse=recurse)
        yield from self.backbone.selectors.parameters(recurse=recurse) # predictor parameter 추가
        yield from self.aggregator.parameters(recurse=recurse)
    
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
    def loss_function(self, s_desc, t_desc, labels, s_out_pred_prob, log_accuracy: bool=False):
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
            
        distill_loss = nn.functional.mse_loss(s_desc, t_desc)

        pred_loss = 0.0
        ratio = self.backbone.keep_ratio
        for i, score in enumerate(s_out_pred_prob):
            pos_ratio = score.mean(dim=1)
            pred_loss = pred_loss + ((pos_ratio - ratio[i]) ** 2).mean()

        loss = loss + distill_loss + pred_loss
    
        return loss
    
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

        s_desc, t_desc, _, _, s_prev_decision, s_out_pred_prob = self(images)

        if torch.isnan(s_desc).any():
            raise ValueError('NaNs in descriptors (student descriptor)')
        if torch.isnan(t_desc).any():
            raise ValueError('NaNs in descriptors (teacher descriptor)')

        # Call the loss_function we defined above
        loss = self.loss_function(s_desc, t_desc, labels, s_out_pred_prob, log_accuracy=True) 
        self.log('loss', loss.item(), logger=True, prog_bar=True)
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