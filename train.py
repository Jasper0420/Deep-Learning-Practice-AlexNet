import os
import sys
import json
import torch
import time
import torch.nn as nn
from torchvision import transforms, datasets, utils
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from tqdm import tqdm
from model import AlexNet


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "val": transforms.Compose([transforms.Resize((224, 224)),  # cannot 224, must (224, 224)
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

    data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
    image_path = os.path.join(data_root, "data_set", "flower_data")  # flower data set path
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)

    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 64
    # nw = 0 #windows训练时设置为0
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  #  lunix number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=4, shuffle=False,
                                                  num_workers=nw)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))
    # test_data_iter = iter(validate_loader)
    # test_image, test_label = test_data_iter.next()
    #
    # def imshow(img):
    #     img = img / 2 + 0.5  # unnormalize
    #     npimg = img.numpy()
    #     plt.imshow(np.transpose(npimg, (1, 2, 0)))
    #     plt.show()
    #
    # print(' '.join('%5s' % cla_dict[test_label[j].item()] for j in range(4)))
    # imshow(utils.make_grid(test_image))

    net = AlexNet(num_classes=5, init_weights=True)# 实例化网络（输出类型为5，初始化权重）
    net.to(device)# 分配网络到指定的设备（GPU/CPU）训练
    loss_function = nn.CrossEntropyLoss()# 交叉熵损失
    # pata = list(net.parameters())
    optimizer = optim.Adam(net.parameters(), lr=0.0002)# 优化器（训练参数，学习率）
    
    epochs = 10
    save_path = './AlexNet.pth'
    best_acc = 0.0
    train_steps = len(train_loader)
    #训练集
    for epoch in range(epochs):
        # train
        net.train()# 训练过程中开启 Dropout
        running_loss = 0.0 #每个 epoch 都会对 running_loss  清零
        time_start = time.perf_counter()	# 对训练一个 epoch 计时
        train_bar = tqdm(train_loader, file=sys.stdout)# 对训练一个 epoch 计时
        for step, data in enumerate(train_bar): # 遍历训练集，step从0开始计算
            images, labels = data   # 获取训练集的图像和标签
            optimizer.zero_grad()	# 清除历史梯度
            outputs = net(images.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss = loss.requires_grad_(True)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            # 打印训练进度（使训练过程可视化）
            rate = (step + 1) / len(train_loader)           # 当前进度 = 当前step / 训练一轮epoch所需总step
            a = "*" * int(rate * 50)
            b = "." * int((1 - rate) * 50)
            print("\rtrain loss: {:^3.0f}%[{}->{}]{:.3f}".format(int(rate * 100), a, b, loss), end="")
            print()
            print('%f s' % (time.perf_counter()-time_start))


          

        # validate
        net.eval()# 验证过程中关闭 Dropout
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]# 以output中值最大位置对应的索引（标签）作为预测输出
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))
        # 保存准确率最高的那次网络参数
        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

    print('Finished Training')
    


if __name__ == '__main__':
    main()