# /usr/bin/env python

import vidmodex
from vidmodex.models import ViViT
from vidmodex.models import MoViNet
from vidmodex.models import SwinT
from vidmodex.generator import Tgan
import vidmodex.config_args
from vidmodex.train import train_datafree

def blackbox_main(data_config):
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

    use_cuda = not config_args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(config_args.seed)
    torch.cuda.manual_seed(config_args.seed)
    np.random.seed(config_args.seed)
    random.seed(config_args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda:%d" % config_args.device if use_cuda else "cpu")
    kwconfig_args = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    global file
    model_dir = "checkpoint/student_{}".format(config_args.model_id)
    config_args.model_dir = model_dir
    if (not os.path.exists(model_dir)):
        os.makedirs(model_dir)

    file = open("{}/logs.txt".format(config_args.model_dir), "w")

    print(config_args)

    config_args.device = device

    config_args.normalization_coefs = None
    config_args.G_activation = torch.tanh



    config_args.num_classes = data_config["num_classes"]
    teacher = Victim()
    teacher.load_state_dict(torch.load(
        data_config["teacher_weight"],
        map_location=device))
    teacher.cuda()
   
    teacher.eval()
    teacher = teacher.to(device)
    print("\n\t\tTraining with {config_args.model} as a Target\n".format)
    correct = 0

    student = Student(224, 16, data_config["num_classes"], 16)


    generator = Generator().cuda()
    generator = generator.to(device)

    config_args.generator = generator
    config_args.student = student
    config_args.teacher = teacher

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
    writer = SummaryWriter(f"runs/BB")
    step = 0
    for epoch in range(1, number_epochs + 1):
        if config_args.scheduler != "none":
            scheduler_S.step()
            scheduler_G.step()

       

        loss_S, loss_G = train_datafree(config_args, teacher=teacher, student=student, generator=generator,
                               device=device, optimizer=[optimizer_S, optimizer_G], epoch=epoch)
        writer.add_scalar("Loss Student", loss_S, global_step=step)
        writer.add_scalar("Loss Generator", loss_G, global_step=step)

        acc, test_loss = test(config_args, student=student, generator=generator,
                   device=device, test_loader=test_loader, epoch=epoch)

        acc_list.append(acc)
        
    
        if acc > best_acc:
            best_acc = acc
            name = 'swinT'
            torch.save(student.state_dict(
            ), f"checkpoint/student_kinetics-{name}.pth")
            torch.save(generator.state_dict(
            ), f"checkpoint/student_kinetics-generator.pth")
        if config_args.store_checkpoints:
            torch.save(student.state_dict(), config_args.log_dir +
                        f"/checkpoint/student.pth")
            torch.save(generator.state_dict(), config_args.log_dir +
                        f"/checkpoint/generator.pth")
            writer.add_scalar("Testing Acc", acc, global_step=step)
            writer.add_scalar("Testing Loss", test_loss, global_step=step)
            step += 1

    print("Best Acc=%.6f" % best_acc)

if __name__ == "__main__":
    Victim    = SwinT
    Student   = ViViT
    Generator = Tgan

    #ToDo: Add ABSL for all the config args
    custom_config = {}
    custom_config["num_classes"] = 400

    blackbox_main(custom_config)
