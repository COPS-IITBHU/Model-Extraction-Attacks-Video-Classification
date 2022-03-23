# import argparse
import json
import os
import random
import sys
from pprint import pprint

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(1, './')
from approximate_gradients import *
from gan import VideoGenerator
from my_utils import *
from swint_victim import SwinTransformer3D as VICTIM
from models.ViViT import *
import config_args

os.environ["CUDA_VISIBLE_DEVICES"] = config_args.cuda_device
sys.path.insert(1, '../pythonProject/')

def myprint(a):
    """Log the print statements"""
    global file
    print(a)
    file.write(a)
    file.write("\n")
    file.flush()

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


def train(config_args, teacher, student, generator, device, optimizer, epoch):
    """Main Loop for one epoch of Training Generator and Student"""

    teacher.eval()
    student.train()


    optimizer_S, optimizer_G = optimizer

    gradients = []

    for i in range(config_args.epoch_itrs):
        """Repeat epoch_itrs times per epoch"""
        for _ in range(config_args.g_iter):
            # Sample Random Noise
            # torch.rand((batch_size, 100), device='cuda')*2-1
            z = torch.randn((config_args.batch_size, config_args.nz), device=device)
            optimizer_G.zero_grad()
            generator.train()
            # Get fake image from generator
            # pre_x returns the output of G before applying the activation

            fake = generator(z)  # torch.rand((batch_size, NOISE_SIZE), device='cuda')*2-1

            # 16, 3, 16, 64, 64
            # APPROX GRADIENT
            approx_grad_wrt_x, loss_G = estimate_gradient_objective(config_args, teacher, student, fake,
                                                                    epsilon=config_args.grad_epsilon, m=config_args.grad_m,
                                                                    num_classes=config_args.num_classes,
                                                                    device=device, pre_x=True)
            # m = 'Number of steps to approximate the gradients'
            fake.backward(approx_grad_wrt_x)
            # print(fake.shape)
            optimizer_G.step()

            if i == 0 and config_args.rec_grad_norm:
                x_true_grad = measure_true_grad_norm(config_args, fake)

        for _ in range(config_args.d_iter):
            z = torch.randn((config_args.batch_size, config_args.nz)).to(device)
            fake = generator(z).detach()
            optimizer_S.zero_grad()

            with torch.no_grad():
                t_logit = teacher(fake)

            # Correction for the fake logits
            if config_args.loss == "l1" and config_args.no_logits:
                t_logit = F.log_softmax(t_logit, dim=1).detach()
                if config_args.logit_correction == 'min':
                    t_logit -= t_logit.min(dim=1).values.view(-1, 1).detach()
                elif config_args.logit_correction == 'mean':
                    t_logit -= t_logit.mean(dim=1).view(-1, 1).detach()

            s_logit = student(fake.permute(0,2,1,3,4))

            loss_S = student_loss(config_args, s_logit, t_logit)
            loss_S.backward()
            optimizer_S.step()

        # Log Results,
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

        # update query budget
        config_args.query_budget -= config_args.cost_per_iteration

        torch.save(generator, "swinT_viviT_weights/gen_model_%d.pt" % i)
        torch.save(student, "swinT_viviT_weights/student_model_%d.pt" % i)
        torch.save(teacher,  "swinT_viviT_weights/teacher_model_%d.pt" % i)

        if config_args.query_budget < config_args.cost_per_iteration:
            return loss_S, loss_G
    return loss_S, loss_G


# vidit bhaiya ho kya? kahne gaya tha. good


def test(config_args, student=None, generator=None, device="cuda", test_loader=None, epoch=0):
    student.eval()
    generator.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = student(data.permute(0, 2, 1, 3, 4))

            # sum up batch loss
            test_loss += F.cross_entropy(output,
                                         target, reduction='sum').item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Average loss: %f{}, Accuracy: {}/{} %f{}\n'.format(
        test_loss, correct, len(test_loader.dataset),
        accuracy))

    with open(config_args.log_dir + "/accuracy.csv", "a") as f:
        f.write("%d,%f\n" % (epoch, accuracy))
    acc = correct / len(test_loader.dataset)
    return acc, test_loss


def compute_grad_norms(generator, student):
    G_grad = []
    for n, p in generator.named_parameters():
        if "weight" in n:
            # print('===========\ngradient{}\n----------\n{}'.format(n, p.grad.norm().to("cpu")))
            G_grad.append(p.grad.norm().to("cpu"))

    S_grad = []
    for n, p in student.named_parameters():
        if "weight" in n:
            # print('===========\ngradient{}\n----------\n{}'.format(n, p.grad.norm().to("cpu")))
            S_grad.append(p.grad.norm().to("cpu"))
    return np.mean(G_grad), np.mean(S_grad)



def singlevideotest(model, video_path):
    model()
    model.cuda()
    with torch.no_grad():
        inp = video2img(video_path).cuda()
        print(f"Expected output: {dic[video_path.split('/')[-2]]}")
        # The top 5 classes
        print(f"Prediction is : {torch.config_argsort(model(inp), 1, True)[:, :5]}")


def main():

    config_args.query_budget *= 10 ** 6
    config_args.query_budget = int(config_args.query_budget)
    if config_args.MAZE:
        print("\n" * 2)
        print("#### /!\ OVERWRITING ALL PARAMETERS FOR MAZE REPLCIATION ####")
        print("\n" * 2)
        config_args.scheduer = "cosine"
        config_args.loss = "kl"
        config_args.batch_size = 16
        config_args.g_iter = 1
        config_args.d_iter = 5
        config_args.grad_m = 10
        config_args.lr_G = 1e-4
        config_args.lr_S = 1

    pprint(config_args, width=80)
    print(config_args.log_dir)
    os.makedirs(config_args.log_dir, exist_ok=True)

    if config_args.store_checkpoints:
        os.makedirs(config_args.log_dir + "/checkpoint", exist_ok=True)

    # Save JSON with parameters
    # with open(config_args.log_dir + "/parameters.json", "w") as f:
    #     json.dump(vars(config_args), f)

    # with open(config_args.log_dir + "/loss.csv", "w") as f:
    #     f.write("epoch,loss_G,loss_S\n")

    # with open(config_args.log_dir + "/accuracy.csv", "w") as f:
    #     f.write("epoch,accuracy\n")

    # if config_args.rec_grad_norm:
    #     with open(config_args.log_dir + "/norm_grad.csv", "w") as f:
    #         f.write("epoch,G_grad_norm,S_grad_norm,grad_wrt_X\n")

    # with open("latest_experiments.txt", "a") as f:
    #     f.write(config_args.log_dir + "\n")
    use_cuda = not config_args.no_cuda and torch.cuda.is_available()

    # Prepare the environment
    torch.manual_seed(config_args.seed)
    torch.cuda.manual_seed(config_args.seed)
    np.random.seed(config_args.seed)
    random.seed(config_args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda:%d" % config_args.device if use_cuda else "cpu")
    kwconfig_args = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # Preparing checkpoints for the best Student
    global file
    model_dir = "checkpoint/student_{}".format(config_args.model_id)
    config_args.model_dir = model_dir
    if (not os.path.exists(model_dir)):
        os.makedirs(model_dir)
    # with open("{}/model_info.txt".format(model_dir), "w") as f:
    #     json.dump(config_args.__dict__, f, indent=2)
    file = open("{}/logs.txt".format(config_args.model_dir), "w")

    print(config_args)

    config_args.device = device
    kinetics400_dataset = videosDataset(
        csv_file="/DATA/shorya/experiment/datafree-model-extraction/dfme/test.csv",
        root_dir="/DATA/shorya/experiment/datafree-model-extraction/dfme/data/")

    # Eigen values and vectors of the covariance matrix
    # _, test_loader = get_dataloader(config_args)
    test_loader = torch.utils.data.DataLoader(
        dataset=kinetics400_dataset, batch_size=16, pin_memory=True, num_workers=1)

    config_args.normalization_coefs = None
    config_args.G_activation = torch.tanh

    num_classes = 400

    # num_classes = 10 if config_args.dataset in ['cifar10', 'svhn'] else 100
    config_args.num_classes = num_classes
    teacher = VICTIM()
    teacher.load_state_dict(torch.load(
        '/DATA/shorya/experiment/datafree-model-extraction/dfme/weights/swint_victim_pretrained.pth',
        map_location=device))
    # teacher.eval()
    teacher.cuda()
    # video_path = "/DATA/shorya/experiment/datafree-model-extraction/dfme/data/kinetics600/rolling pastry/dpRdeJP2juU_000002_000012.mp4"
    # with torch.no_grad():
    #     inp = video2img(video_path).cuda()
    #     print(f"Expected output: {dic[video_path.split('/')[-2]]}")
    #     # print(inp.shape)
    #     LABEL = torch.config_argsort(teacher(inp), 1, True)[:, :40]
    # print(LABEL)
    teacher.eval()
    teacher = teacher.to(device)
    # myprint("Teacher restored from %movinets" % (config_args.ckpt))
    print("\n\t\tTraining with {config_args.model} as a Target\n".format)
    correct = 0
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = teacher(data)
            # print(output.shape)
            # print(data.shape)
            # print(output)
            # print(torch.sum(output, dim=1))
            # break
            # get the index of the max log-probability
            pred = output.argmax(dim=1)
            correct += pred.eq(target.view_as(pred)).sum().item()
            # print(pred)
            # print(target)
    accuracy = 100. * correct / len(test_loader.dataset)
    print('\nTeacher - Test set: Accuracy: {}/{} ({:.4f}%)\n'.format(correct,
                                                                     len(test_loader.dataset), accuracy))
    # student = get_classifier(config_args.student_model, pretrained=False, num_classes=config_args.num_classes)
    student = ViViT(224, 16, 400, 16)

    # generator = network.gan.GeneratorA(nz=config_args.nz, nc=3, img_size=32, activation=config_args.G_activation)

    generator = VideoGenerator().cuda()
    generator.load_state_dict(torch.load('state_normal81000.ckpt')['model_state_dict'][0])
    # student = student.to(device)
    generator = generator.to(device)

    config_args.generator = generator
    config_args.student = student
    config_args.teacher = teacher

    # if config_args.student_load_path :
    #     # "checkpoint/student_no-grad/cifar10-resnet34_8x.pt"
    #     student.load_state_dict( torch.load( config_args.student_load_path ) )
    #     myprint("Student initialized from %movinets"%(config_args.student_load_path))
    #     acc = test(config_args, student=student, generator=generator, device = device, test_loader = test_loader)

    # Compute the number of epochs with the given query budget:

    config_args.cost_per_iteration = config_args.batch_size * \
                              (config_args.g_iter * (config_args.grad_m + 1) + config_args.d_iter)
    number_epochs = config_args.query_budget // (
            config_args.cost_per_iteration * config_args.epoch_itrs) + 1

    print("\nTotal budget:", {config_args.query_budget // 1000}, "k")
    print("Cost per iterations: ", config_args.cost_per_iteration)
    print("Total number of epochs: ", number_epochs)

    optimizer_S = optim.SGD(student.parameters(
    ), lr=config_args.lr_S, weight_decay=config_args.weight_decay, momentum=0.9)

    if config_args.MAZE:
        optimizer_G = optim.SGD(generator.parameters(
        ), lr=config_args.lr_G, weight_decay=config_args.weight_decay, momentum=0.9)
    else:
        optimizer_G = optim.Adam(generator.parameters(), lr=config_args.lr_G)

    steps = sorted([int(step * number_epochs) for step in config_args.steps])
    print("Learning rate scheduling at steps: ", steps)
    # print()

    if config_args.scheduler == "multistep":
        scheduler_S = optim.lr_scheduler.MultiStepLR(
            optimizer_S, steps, config_args.scale)
        scheduler_G = optim.lr_scheduler.MultiStepLR(
            optimizer_G, steps, config_args.scale)
    elif config_args.scheduler == "cosine":
        scheduler_S = optim.lr_scheduler.CosineAnnealingLR(
            optimizer_S, number_epochs)
        scheduler_G = optim.lr_scheduler.CosineAnnealingLR(
            optimizer_G, number_epochs)

    best_acc = 0
    acc_list = []
    writer = SummaryWriter(f"runs/SwinT/BB")
    step = 0
    for epoch in range(1, number_epochs + 1):
        # Train
        if config_args.scheduler != "none":
            scheduler_S.step()
            scheduler_G.step()

        # writer.add_scalar("Loss Generator", acc, global_step=step)
        # writer.add_scalar("Loss Sgtudent", acc, global_step=step)

        loss_S, loss_G = train(config_args, teacher=teacher, student=student, generator=generator,
                               device=device, optimizer=[optimizer_S, optimizer_G], epoch=epoch)
        writer.add_scalar("Loss Student", loss_S, global_step=step)
        writer.add_scalar("Loss Generator", loss_G, global_step=step)

        acc, test_loss = test(config_args, student=student, generator=generator,
                   device=device, test_loader=test_loader, epoch=epoch)

        acc_list.append(acc)
        
        # if acc > best_acc:
        #     best_acc = acc
        #     name = 'resnet34_8x'
        #     torch.save(student.state_dict(
        #     ), f"checkpoint/student_{config_args.model_id}/{config_args.dataset}-{name}.pt")
        #     torch.save(generator.state_dict(
        #     ), f"checkpoint/student_{config_args.model_id}/{config_args.dataset}-{name}-generator.pt")
    
        if acc > best_acc:
            best_acc = acc
            name = 'swinT'
            torch.save(student.state_dict(
            ), f"checkpoint/student_kinetics400-{name}.pth")
            torch.save(generator.state_dict(
            ), f"checkpoint/student_kinetics400-generator.pth")
        # vp.add_scalar('Acc', epoch, acc)
        if config_args.store_checkpoints:
            torch.save(student.state_dict(), config_args.log_dir +
                        f"/checkpoint/student.pth")
            torch.save(generator.state_dict(), config_args.log_dir +
                        f"/checkpoint/generator.pth")
            writer.add_scalar("Testing Acc", acc, global_step=step)
            writer.add_scalar("Testing Loss", test_loss, global_step=step)
            step += 1

    myprint("Best Acc=%.6f" % best_acc)

    # with open(config_args.log_dir + "/Max_accuracy = %f"%best_acc, "w") as f:
    #     f.write(" ")

    # import csv
    # os.makedirs('log', exist_ok=True)
    # with open('log/DFAD-%movinets.csv'%(config_args.dataset), 'a') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(acc_list)


if __name__ == '__main__':
    # torch.backends.cudnn.benchmark = True
    main()
