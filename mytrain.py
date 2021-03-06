# -------------------------------------#
#       对数据集进行训练
# -------------------------------------#
import os
import numpy as np
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from utils.dataloader import yolo_dataset_collate, YoloDataset
from nets.yolo_training import YOLOLoss, Generator
from nets.yolo4 import YoloBody

from tensorboardX import SummaryWriter

# ---------------------------------------------------#
#   获得类和先验框
# ---------------------------------------------------#
def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape([-1, 3, 2])[::-1, :, :]


def fit_ont_epoch(net, yolo_losses, epoch, epoch_size, epoch_size_val, gen, genval, Epoch, cuda):
    total_loss = 0
    val_loss = 0
    start_time = time.time()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_size:
            break
        images, targets = batch[0], batch[1]
        with torch.no_grad():
            if cuda:
                images = Variable(torch.from_numpy(images).type(torch.FloatTensor)).cuda()
                targets = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets]
            else:
                images = Variable(torch.from_numpy(images).type(torch.FloatTensor))
                targets = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets]
        optimizer.zero_grad()
        outputs = net(images)
        losses = []
        for i in range(3):
            loss_item = yolo_losses[i](outputs[i], targets)
            losses.append(loss_item[0])
        loss = sum(losses)
        loss.backward()
        optimizer.step()

        # 将训练loss写入到tensorboard
        # writer.add_scalar('Train_loss', loss, (epoch * epoch_size + iteration))

        total_loss += loss
        waste_time = time.time() - start_time
        print('\nEpoch:' + str(epoch + 1) + '/' + str(Epoch))
        print('iter:' + str(iteration) + '/' + str(epoch_size) + ' || Total Loss: %.4f || %.4fs/step' % (
        total_loss / (iteration + 1), waste_time))
        start_time = time.time()

    # writer.add_scalar('Train_loss', total_loss / (iteration + 1), (epoch * epoch_size + iteration))
    # writer.add_scalar('Train_loss', total_loss / (iteration + 1), epoch)
    writer.add_scalar('Train_loss', total_loss / (epoch_size + 1), epoch)

    print('Start Validation')
    for iteration, batch in enumerate(genval):
        if iteration >= epoch_size_val:
            break
        images_val, targets_val = batch[0], batch[1]

        with torch.no_grad():
            if cuda:
                images_val = Variable(torch.from_numpy(images_val).type(torch.FloatTensor)).cuda()
                targets_val = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets_val]
            else:
                images_val = Variable(torch.from_numpy(images_val).type(torch.FloatTensor))
                targets_val = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets_val]
            optimizer.zero_grad()
            outputs = net(images_val)
            losses = []
            for i in range(3):
                loss_item = yolo_losses[i](outputs[i], targets_val)
                losses.append(loss_item[0])
            loss = sum(losses)
            val_loss += loss

            # 将loss写入到tensorboard
    # writer.add_scalar('Val_loss', val_loss / (epoch_size_val + 1), (epoch * epoch_size_val + iteration))
    writer.add_scalar('Val_loss', val_loss / (epoch_size_val + 1), epoch)

    print('Finish Validation')
    print('\nEpoch:' + str(epoch + 1) + '/' + str(Epoch))
    print('Total Loss: %.4f || Val Loss: %.4f ' % (total_loss / (epoch_size + 1), val_loss / (epoch_size_val + 1)))

    if (epoch + 1) % 5 == 0:
        torch.save(model.state_dict(), 'logs/Epoch%d-Total_Loss%.4f-Val_Loss%.4f.pth' % (
             (epoch + 1), total_loss / (epoch_size + 1), val_loss / (epoch_size_val + 1)))

    # print('Saving state, iter:', str(epoch + 1))
    # torch.save(model.state_dict(), 'logs/Epoch%d-Total_Loss%.4f-Val_Loss%.4f.pth' % (
    # (epoch + 1), total_loss / (epoch_size + 1), val_loss / (epoch_size_val + 1)))

    output = open('./model_data/loss.txt', 'a')
    output.write(str(float(total_loss / (epoch_size + 1))) + ' ' + str(float(val_loss / (epoch_size_val + 1))))
    output.write('\n')
    output.close()


if __name__ == "__main__":

    # -------------------------------#
    #   输入的shape大小
    #   显存比较小可以使用416x416
    #   显存比较大可以使用608x608
    # -------------------------------#
    input_shape = (416, 416)
    # -------------------------------#
    #   tricks的使用设置
    # -------------------------------#
    Cosine_lr = True
    mosaic = True
    # 用于设定是否使用cuda
    Cuda = True
    smoooth_label = 0

    Use_Data_Loader = True

    annotation_path = './data/dataset/train.txt'
    classes_path = './data/classes/yolov4_classes.txt'
    anchors_path = './data/anchors/yolov4_anchors.txt'
    class_names = get_classes(classes_path)
    anchors = get_anchors(anchors_path)
    num_classes = len(class_names)

    # 创建模型
    model = YoloBody(len(anchors[0]), num_classes)
    model_path = "./model_data/yolo4_weights.pth"

    print('Loading weights into state dict...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path, map_location=device)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print('Finished!')

    net = model.train()

    if Cuda:
        net = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        net = net.cuda()

    # 建立loss函数
    yolo_losses = []
    for i in range(3):
        yolo_losses.append(YOLOLoss(np.reshape(anchors, [-1, 2]), num_classes, \
                                    (input_shape[1], input_shape[0]), smoooth_label, Cuda))

    # 0.1用于验证，0.9用于训练
    val_split = 0.01
    with open(annotation_path) as f:
        lines = f.readlines()      # 读取标注文件中每行
    np.random.seed(10101)
    np.random.shuffle(lines)      # 将其顺序打乱
    np.random.seed(None)
    num_val = int(len(lines) * val_split)  # 行数 * 0.1  验证集个数
    num_train = len(lines) - num_val       # 剩下是训练集个数

    writer = SummaryWriter(log_dir='logs', flush_secs=60)
    graph_inputs = torch.from_numpy(np.random.rand(1,3,input_shape[0],input_shape[1])).type(torch.FloatTensor).cuda()
    writer.add_graph(model, (graph_inputs,))

    if True:
        lr = 1e-3
        Batch_size = 8
        Init_Epoch = 0
        Freeze_Epoch = 25

        optimizer = optim.Adam(net.parameters(), lr, weight_decay=5e-4)
        if Cosine_lr:
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
        else:
            lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

        if Use_Data_Loader:
            train_dataset = YoloDataset(lines[:num_train], (input_shape[0], input_shape[1]), mosaic=mosaic)
            val_dataset = YoloDataset(lines[num_train:], (input_shape[0], input_shape[1]), mosaic=False)
            gen = DataLoader(train_dataset, batch_size=Batch_size, num_workers=8, pin_memory=True,
                             drop_last=True, collate_fn=yolo_dataset_collate)
            gen_val = DataLoader(val_dataset, batch_size=Batch_size, num_workers=8, pin_memory=True,
                                 drop_last=True, collate_fn=yolo_dataset_collate)
        else:
            gen = Generator(Batch_size, lines[:num_train],
                            (input_shape[0], input_shape[1])).generate(mosaic=mosaic)
            gen_val = Generator(Batch_size, lines[num_train:],
                                (input_shape[0], input_shape[1])).generate(mosaic=False)

        epoch_size = max(1, num_train // Batch_size)    # 计算训练小周期的个数
        epoch_size_val = num_val // Batch_size          # 计算验证小周期的个数
        # ------------------------------------#
        #   冻结一定部分训练
        # ------------------------------------#
        for param in model.backbone.parameters():
            param.requires_grad = False       # 模型的主干部分

        RESUME = False
        if RESUME:
            path_checkpoint = "./checkpoint/ckpt_qian_0.pth"  # 断点路径
            checkpoint = torch.load(path_checkpoint)  # 加载断点

            model.load_state_dict(checkpoint['net'])  # 加载模型可学习参数

            optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
            start_epoch = checkpoint['epoch']  # 设置开始的epoch
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

        for epoch in range(Init_Epoch, Freeze_Epoch):
            fit_ont_epoch(net, yolo_losses, epoch, epoch_size, epoch_size_val, gen, gen_val, Freeze_Epoch, Cuda)
            lr_scheduler.step()

            # if epoch % 5 == 0:
            #     print('learning rate:', optimizer.state_dict()['param_groups'][0]['lr'])
            #     checkpoint = {
            #         "net": model.state_dict(),
            #         'optimizer': optimizer.state_dict(),
            #         "epoch": epoch,
            #         'lr_scheduler': lr_scheduler.state_dict()
            #     }
            #     if not os.path.isdir("./checkpoint"):
            #         os.mkdir("./checkpoint")
            #     torch.save(checkpoint, './checkpoint/ckpt_qian_%s.pth' % (str(epoch)))

    if True:
        lr = 1e-3
        Batch_size = 2
        Freeze_Epoch = 25
        Unfreeze_Epoch = 300

        optimizer = optim.Adam(net.parameters(), lr, weight_decay=5e-4)
        if Cosine_lr:
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
        else:
            lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

        if Use_Data_Loader:
            train_dataset = YoloDataset(lines[:num_train], (input_shape[0], input_shape[1]), mosaic=mosaic)
            val_dataset = YoloDataset(lines[num_train:], (input_shape[0], input_shape[1]), mosaic=False)
            gen = DataLoader(train_dataset, batch_size=Batch_size, num_workers=8, pin_memory=True,
                             drop_last=True, collate_fn=yolo_dataset_collate)
            gen_val = DataLoader(val_dataset, batch_size=Batch_size, num_workers=8, pin_memory=True,
                                 drop_last=True, collate_fn=yolo_dataset_collate)
        else:
            gen = Generator(Batch_size, lines[:num_train],
                            (input_shape[0], input_shape[1])).generate(mosaic=mosaic)
            gen_val = Generator(Batch_size, lines[num_train:],
                                (input_shape[0], input_shape[1])).generate(mosaic=False)

        epoch_size = max(1, num_train // Batch_size)
        epoch_size_val = num_val // Batch_size
        # ------------------------------------#
        #   解冻后训练
        # ------------------------------------#
        for param in model.backbone.parameters():
            param.requires_grad = True

        RESUME = False
        if RESUME:
            path_checkpoint = "./checkpoint/ckpt_hou_199.pth"  # 断点路径
            checkpoint = torch.load(path_checkpoint)  # 加载断点

            model.load_state_dict(checkpoint['net'])  # 加载模型可学习参数

            optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
            start_epoch = checkpoint['epoch']  # 设置开始的epoch
            lr_schedule.load_state_dict(checkpoint['lr_schedule'])

        for epoch in range(Freeze_Epoch, Unfreeze_Epoch):
            fit_ont_epoch(net, yolo_losses, epoch, epoch_size, epoch_size_val, gen, gen_val, Unfreeze_Epoch, Cuda)
            lr_scheduler.step()

            # if epoch % 20 == 0:
            #     print('epoch:', epoch)
            #     print('learning rate:', optimizer.state_dict()['param_groups'][0]['lr'])
            #     checkpoint = {
            #         "net": model.state_dict(),
            #         'optimizer': optimizer.state_dict(),
            #         "epoch": epoch,
            #         'lr_scheduler': lr_scheduler.state_dict()
            #     }
            #     if not os.path.isdir("./checkpoint"):
            #         os.mkdir("./checkpoint")
            #     torch.save(checkpoint, './checkpoint/ckpt_hou_%s.pth' % (str(epoch)))




