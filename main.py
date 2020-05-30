# -*- coding: utf-8 -*-
import argparse  # 命令行解释器相关程序，命令行解释器
import random
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn  # gpu 使用
import torch.distributed as dist  # 分布式
import torch.optim  # 优化器
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
# from torchnet.logger import MeterLogger
# import transforms
import torchvision.datasets as datasets
import torchvision.models
from models.model import *
from utils.utils import *
# from lr_scheduler import *
# from torch.utils.tensorboard import SummaryWriter

# 解析器中的参数一般不用修改，常用参数保存在 config.py 中
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')  # description 在帮助文档显示之前出现的名称
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
parser.add_argument('--world-size', default=-1, type=int, help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int, help='node rank for distributed training')
parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

best_acc1 = 0


def main():
    args = parser.parse_args()  # 解析参数
    # 设置随机种子
    random.seed(config.seed)  # 随机seed保持一致
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)  # 为CPU设置种子用于生成随机数
    torch.cuda.manual_seed_all(config.seed)  # 为所有的GPU设置种子
    cudnn.deterministic = True  # 网络前馈结果保持一致，但拖慢训练速度

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()  # 返回可得到的GPU数量。
    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1, model
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    # mlog = MeterLogger(server="localhost", port=8097, nclass=45, title="Remot classfication")

    # ---------------------------- 创建模型 -------------------------------------
    # model = get_net(pre=True)
    if config.model_names == 'alexnet':
        model = torchvision.models.alexnet(pretrained=True)
        model.fc = torch.nn.Linear(4096, config.num_classes)
        print("Useing Pre-training Model:", config.model_names)
        print(config.model_names, "最后一层输入节点数:", 4096)
    elif config.model_names == 'vgg16':
        model = torchvision.models.vgg16(pretrained=True)
        model.fc = torch.nn.Linear(4096, config.num_classes)
        print(config.model_names, "最后一层输入节点数:", 4096)
    elif config.model_names == 'resnet50':
        model = torchvision.models.resnet50(pretrained=True)
        fc_features = model.fc.in_features  # 提取最后一层得输入节点数
        model.fc = torch.nn.Linear(fc_features, config.num_classes)
        print(config.model_names, "最后一层输入节点数:", model.fc.in_features)
    elif config.model_names == 'googlenet':
        model = torchvision.models.googlenet(pretrained=True)
        fc_features = model.fc.in_features
        model.fc = torch.nn.Linear(fc_features, config.num_classes)
        print(config.model_names, "最后一层输入节点数:", model.fc.in_features)
    else:
        print("未设置需要的预训练模型，请在'main.py'中'创建模型部分添加所需模型'")
    print("Fine-tuned the number of 'Output nodes'in the last layer to '%s' of the '%s' pre-training model" %
          (config.num_classes, config.model_names))
    print("Input numeric parameters for network"
          "\nImage size:%s"
          "\nEpochs:%s"
          "\nBatch size:%s"
          "\nLearning rate:%s"
          "\nStep size:%s"
          "\nPrint frequency:%s"
          "\nNumber classes:%s"
          % (config.image_size, config.epochs, config.batch_size, config.lr,
             config.step_size,config.print_freq, config.num_classes))
    # 1、# Xavier init
    # model = init_weights(model)
    # print("the weights initilized with Xavier algorithm")
    # --------------------------- 分布式训练 -----------------------------------
    if args.distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            config.batch_size = int(config.batch_size / ngpus_per_node)
            config.workers = int((config.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        if config.model_names.startswith('alexnet') or config.model_names.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # 让内置的 cuDNN 的 auto-tuner 自动寻找最适合当前配置的高效算法，来达到优化运行效率
    cudnn.benchmark = True

    traindir = os.path.join(config.data, 'train')
    print(traindir)
    valdir = os.path.join(config.data, 'val')
    print(valdir)

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.Resize((224, 224)),  # 重置图像分辨率
            # transforms.RandomResizedCrop(config.image_size),  # 随机长宽比裁剪
            # transforms.RandomHorizontalFlip(),  # 水平翻转，默认值0.5
            # transforms.ColorJitter(brightness=0.4, saturation=0.4, hue=0.4),  # 修改亮度、对比度和饱和度
            # transforms.RandomErasing(),  # 6、random erasing(does not work)
            # transforms.CutOut(56),  # 7、cutout (does not work)
            transforms.ToTensor(),  # 将PIL Image或者 ndarray 转换为tensor
            transforms.Normalize(mean=config.train_mean, std=config.train_std),
        ]))

    if args.distributed:
        # 采样器可以约束数据加载进数据集的子集
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,  # default (train_sampler is None)
        num_workers=config.workers,
        pin_memory=True,
        sampler=train_sampler)

    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize((224, 224)),
            # transforms.ToCVImage(),
            # transforms.CenterCrop(config.image_size),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.val_mean, std=config.val_std),
        ]))

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.workers,
        pin_memory=True)

    # ----------------定义损失函数和优化器--------------------------------------
    # 4、label smoothing
    # criterion = LSR()

    # criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    criterion = CrossEntropyLabelSmooth(num_classes=config.num_classes, epsilon=0.1).cuda(args.gpu)

    # 3、apply no weight decay on bias
    # params = split_weights(model)
    # optimizer = torch.optim.SGD(params,config.lr,momentum=config.momentum,
    #             weight_decay=config.weight_decay,nesterov=True)

    params = model.parameters()
    optimizer = torch.optim.SGD(params, config.lr, momentum=config.momentum, weight_decay=config.weight_decay)

    # optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, amsgrad=True, weight_decay=config.weight_decay)

    # 2、set up warmup phase learning rate scheduler
    # iter_per_epoch = len(train_loader)
    # warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * config.warm)

    # # 5、set up training phase learning rate scheduler(defau：batchsize 256, lr 0.04)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.milestones)

    # 8、cosine learning rate decay(does not work)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.epochs - config.warm)

    # 每过step_size个epoch
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.step_size, gamma=0.1)

    for epoch in range(config.start_epoch, config.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # --------------- 训练模式 --------------------------
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        progress = ProgressMeter(
            len(train_loader),
            [batch_time, data_time, losses, top1, top5],
            prefix="Epoch: [{}]".format(epoch))

        # 切换到train模式
        model.train()
        end = time.time()
        for i, (images, target) in enumerate(train_loader):
            # 测量数据加载时间
            data_time.update(time.time() - end)
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # # 指定对象标签平滑，修改对象时在[ ]中，加入对象标签序号即可
            # target_numpy = target.cpu().numpy()
            # for j in range(config.batch_size):
            #     if target_numpy[j] in [27, 29, 38, 42]:  # palace,railway,sparse_residential,terrace
            #         criterion = CrossEntropyLabelSmooth(num_classes=config.num_classes, epsilon=0.1).cuda(args.gpu)
            #     else:
            #         criterion = nn.CrossEntropyLoss().cuda(args.gpu)

            # 计算输出
            output = model(images)
            loss = criterion(output, target)
            # 测量准确性并记录损失
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # 计算梯度并执行SGD步骤
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step(epoch)
            # if epoch > config.warm:
            #     scheduler.step(epoch)
            # else:
            #     warmup_scheduler.step()

            # 测量时间
            batch_time.update(time.time() - end)
            end = time.time()

            # ----------训练模式下打印频率------------
            if i % config.print_freq == 0:
                progress.display(i)

        # ------------------验证模式 ------------------------------------
        # acc1 = validate(val_loader, model, criterion, args, epoch, writer)
        acc1 = validate(val_loader, model, criterion, args, epoch)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': config.model_names,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
            }, is_best)


if __name__ == '__main__':
    main()
