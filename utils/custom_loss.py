import torch.nn.functional as F

def student_loss(config_args, s_logit, t_logit, return_t_logits=False):
    """Kl/ L1 Loss for student"""
    print_logits = False
    if config_args.loss == "l1":
        loss_fn = F.l1_loss
        loss = loss_fn(s_logit, t_logit.detach())
    elif config_args.loss == "kl":
        loss_fn = F.kl_div
        s_logit = F.log_softmax(s_logit, dim=1)  # log - Probability
        t_logit = F.softmax(t_logit, dim=1)  # Probality
        loss = loss_fn(s_logit, t_logit.detach(), reduction="batchmean")
    else:
        raise ValueError(config_args.loss)

    if return_t_logits:
        return loss, t_logit.detach()
    else:
        return loss


def generator_loss(config_args, s_logit, t_logit, z=None, z_logit=None, reduction="mean"):
    assert 0

    loss = - F.l1_loss(s_logit, t_logit, reduction=reduction)

    return loss
