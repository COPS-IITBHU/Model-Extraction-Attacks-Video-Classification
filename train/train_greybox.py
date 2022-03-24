import torch
import numpy as np
import pandas as pd
import torchvision
import os
import tqdm
from vidmodex.optim.approximate_gradients import estimate_gradient_objective

def train_Wdata(config_args, teacher, student, generator, dataloader, device, optimizer, epoch):
    """Main Loop for one epoch of Training Generator and Student"""

    teacher.eval()
    student.train()


    optimizer_S, optimizer_G = optimizer

    gradients = []

    for i in range(config_args.epoch_itrs):
        """Repeat epoch_itrs times per epoch"""
        for j, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            z = torch.randn((config_args.batch_size, config_args.nz), device=device)
            optimizer_G.zero_grad()
            generator.train()

            fake = generator(z)  


           
            approx_grad_wrt_x, loss_G = estimate_gradient_objective(config_args, teacher, student, fake,
                                                                    epsilon=config_args.grad_epsilon, m=config_args.grad_m,
                                                                    num_classes=config_args.num_classes,
                                                                    device=device, pre_x=True)
            fake.backward(approx_grad_wrt_x)
            optimizer_G.step()

            if i == 0 and config_args.rec_grad_norm:
                x_true_grad = measure_true_grad_norm(config_args, fake)
            
            mask = torch.random.rand(config_args.batch_size) > config_args.explo_thres

            out = mask * data + (1-mask) * fake.detach()

            optimizer_S.zero_grad()

            with torch.no_grad():
                t_logit = teacher(out)

            if config_args.loss == "l1" and config_args.no_logits:
                t_logit = F.log_softmax(t_logit, dim=1).detach()
                if config_args.logit_correction == 'min':
                    t_logit -= t_logit.min(dim=1).values.view(-1, 1).detach()
                elif config_args.logit_correction == 'mean':
                    t_logit -= t_logit.mean(dim=1).view(-1, 1).detach()

            s_logit = student(out.permute(0,2,1,3,4))

            loss_S = student_loss(config_args, s_logit, t_logit)
            loss_S.backward()
            optimizer_S.step()

        if i % config_args.log_interval == 0:
            print('Train Epoch:', epoch, "[", i, '/', config_args.epoch_itrs, "(", 100 * float(i) / float(
                config_args.epoch_itrs), "%)]\tG_Loss: ", loss_G.item(), "S_loss: ", loss_S.item())
            if i == 0:
                with open(config_args.log_dir + "/loss.csv", "a") as f:
                    f.write("%d,%f,%f\n" % (epoch, loss_G, loss_S))

            if config_args.rec_grad_norm and i == 0:

                G_grad_norm, S_grad_norm = compute_grad_norms(
                    generator, student)
                if i == 0:
                    with open(config_args.log_dir + "/norm_grad.csv", "a") as f:
                        f.write("%d,%f,%f,%f\n" %
                                (epoch, G_grad_norm, S_grad_norm, x_true_grad))

        config_args.query_budget -= config_args.cost_per_iteration

        torch.save(generator, "weights/gen_model_%d.pt" % i)
        torch.save(student, "weights/student_model_%d.pt" % i)
        torch.save(teacher,  "weights/teacher_model_%d.pt" % i)

        if config_args.query_budget < config_args.cost_per_iteration:
            return loss_S, loss_G
    return loss_S, loss_G
