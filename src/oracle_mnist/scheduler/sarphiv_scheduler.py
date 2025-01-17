import torch


def get_schedular(
    optimizer, lr_half_period, lr_mult_period, lr_min, lr_warmup_max, lr_warmup_period
):
    lr_cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer=optimizer,
        T_0=int(lr_half_period),
        T_mult=int(lr_mult_period),
        eta_min=float(lr_min),
    )
    lr_super = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        max_lr=float(lr_warmup_max),
        total_steps=int(lr_warmup_period),
    )
    lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer=optimizer,
        schedulers=[lr_super, lr_cosine],  # type: ignore
        milestones=[int(lr_warmup_period)],
    )

    return lr_scheduler
