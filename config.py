"""
常用配置信息
"""
from datetime import datetime


class DefaultConfigs(object):
    # 1.string parameters
    model_names = "alexnet"  # [alexnet, vgg16, resnet, googlenet]
    data = r"D:\Project\2020\data\NWPU45"
    LOG_DIR = "runs"
    time_now = "{}".format(datetime.now().isoformat())
    image_size = 224  # input image size for network
    milestones = [300, 350, 400]
    # train_mean = [0.48560741861744905, 0.49941626449353244, 0.43237713785804116]
    # train_std = [0.2321024260764962, 0.22770540015765814, 0.2665100547329813]
    # val_mean = [0.4862169586881995, 0.4998156522834164, 0.4311430419332438]
    # val_std = [0.23264268069040475, 0.22781080253662814, 0.26667253517177186]
    train_mean = [0.5, 0.5,0.5]
    train_std = [0.5, 0.5, 0.5]
    val_mean = [0.5, 0.5, 0.5]
    val_std = [0.5, 0.5, 0.5]
    # 2.numeric parameters
    epochs = 100
    start_epoch = 0
    batch_size = 4  # default 256
    lr = 0.0005  # learning rate default 0.001
    step_size = 30
    warm = 5  # warm up phase
    momentum = 0.9
    workers = 1  # default workers = 4
    seed = 888  # seed for initializing training.
    weight_decay = 1e-4  # weight decay (default: 1e-4)
    print_freq = 10  # print frequency (default: 10)
    num_classes = 21


config = DefaultConfigs()
