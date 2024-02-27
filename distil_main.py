import torch

import pytorch_lightning as pl
import pytorch_lightning.callbacks as pl_callbacks

from distil_vpr_model import DistillationModel
from dataloaders.GSVCitiesDataloader import GSVCitiesDataModule

from argparse import ArgumentParser

DINOV2_ARCHS = {
    'dinov2_vits14': 384,
    'dinov2_vitb14': 768,
    'dinov2_vitl14': 1024,
    'dinov2_vitg14': 1536,
}

if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    # torch.set_float32_matmul_precision('high')

    parser = ArgumentParser()
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--teacher-config', type=str, required=True)
    parser.add_argument('--tag', type=str, required=True)

    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--check-val', type=int, default=1)
    parser.add_argument('--lr', type=float, default=4e-5)

    args = parser.parse_args()
    
    datamodule = GSVCitiesDataModule(
        batch_size=16,
        img_per_place=4,
        min_img_per_place=4,
        shuffle_all=False, # shuffle all images or keep shuffling in-city only
        random_sample_from_each_place=True,
        image_size=(224, 224),
        num_workers=8,
        persistent_workers=True, # for CPU
        show_data_stats=True,
        val_set_names=[
            # 'pitts30k_val', 
            # 'pitts30k_test', 
            'pitts250k_val', 
            'pitts250k_test', 
            # 'sped', 
            'msls_val', 
            'nordland', 
        ], 
    )
    
    student_arch = 'dinov2_vits14'
    model = DistillationModel(
        #---- Teacher
        teacher_config=args.teacher_config,  # followed by '/lightning_logs/version_0/hparams.yaml'
        
        #---- Student
        student_arch=student_arch,
        student_config={
            'model_name': student_arch,
            'img_size': 154, 
        },
        
        lr = args.lr,
        optimizer='adamW',
        weight_decay=9.5e-9, # 0.001 for sgd and 0 for adam,
        momentum=0.9,
        lr_sched='linear',
        lr_sched_args = {
            'start_factor': 1,
            'end_factor': 0.2,
            'total_iters': 4000,
        },

        #----- Loss functions
        # example: ContrastiveLoss, TripletMarginLoss, MultiSimilarityLoss,
        # FastAPLoss, CircleLoss, SupConLoss,
        loss_name='MultiSimilarityLoss',
        miner_name='MultiSimilarityMiner', # example: TripletMarginMiner, MultiSimilarityMiner, PairMarginMiner
        miner_margin=0.1,
        faiss_gpu=True,
        faiss_device=args.device
    )

    # model params saving using Pytorch Lightning
    # we save the best 3 models accoring to Recall@1 on pittsburg val
    checkpoint_cb = pl_callbacks.ModelCheckpoint(
        monitor='pitts250k_val/R1',
        filename=f'{model.encoder_arch}' + '_({epoch:02d})_R1[{pitts250k_val/R1:.4f}]_R5[{pitts250k_val/R5:.4f}]',
        auto_insert_metric_name=False,
        save_weights_only=True,
        save_top_k=1,
        save_last=True,
        mode='max'
    )

    #------------------
    # we instanciate a trainer
    trainer = pl.Trainer(
        accelerator='gpu', 
        devices=[args.device],
        default_root_dir=f'./logs_distill/{args.tag}', # Tensorflow can be used to viz 
        num_nodes=1,
        num_sanity_val_steps=0, # runs a validation step before stating training
        max_epochs=args.epochs,  # increased by 8 because the batch was halved. 
        check_val_every_n_epoch=args.check_val, # run validation every epoch
        callbacks=[checkpoint_cb],# we only run the checkpointing callback (you can add more)
        reload_dataloaders_every_n_epochs=1, # we reload the dataset to shuffle the order
        log_every_n_steps=20,
    )

    # we call the trainer, we give it the model and the datamodule
    trainer.fit(model=model, datamodule=datamodule)
