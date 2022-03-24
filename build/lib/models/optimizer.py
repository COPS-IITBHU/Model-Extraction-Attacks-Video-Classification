import torch 

## Optimizer config that worked best with MoViNet as Victim and ViViT as student
movinet_w_optimizer = lambda x : torch.optim.AdamW(
    x.parameters(), 
    lr=3e-4, 
    betas=(0.9, 0.999), 
    eps=1e-8, 
    weight_decay=0.02
)

movinet_lr_scheduler = lambda x: torch.optim.lr_scheduler.CosineAnnealingLR(x , T_max=7)

## Optimizer config that worked best with SwinT as Victim and ViViT as student
swint_w_optimizer = lambda x:  torch.optim.Adam(
    x.parameters(),
    lr=0.001,
    betas=(0.9, 0.999),
    weight_decay=0.02
)   