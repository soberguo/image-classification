import os
import sys
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm

from resnet import resnet34,resnext50_32x4d
import visdom

def main(bs,epochs,lr,path):
    batch_size = bs
    data_root=path

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    #data_root = os.path.abspath(os.path.join(os.getcwd(), "./.."))  # get data root path
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


    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))

    net = resnext50_32x4d()
    # load pretrain weights
    # download url: https://download.pytorch.org/models/resnet34-333f7ec4.pth
    model_weight_path = "./resnext50_32x4d-7cdf4587.pth"
    assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
    net.load_state_dict(torch.load(model_weight_path, map_location='cpu'))
    #resnet注释下面两行，resnext不需要注释
    for param in net.parameters():
        param.requires_grad = False



    # change fc layer structure
    in_channel = net.fc.in_features
    net.fc = nn.Linear(in_channel, 5)
    net.to(device)

    # define loss function
    loss_function = nn.CrossEntropyLoss()

    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=lr)

    viz = visdom.Visdom()
    viz.line([0], [0], win='loss', opts=dict(title='loss'))
    viz.line([0], [0], win='val_acc', opts=dict(title='val_acc'))

    best_acc = 0.0
    save_path = './resNext50.pth'
    train_steps = len(train_loader)
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            logits = net(images.to(device))
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)
        viz.line([running_loss/train_steps], [epoch+1], win='loss', update='append',opts=dict(title='val_acc',legend=['acc'],
                                                                                  xtickmin=0,xtickmax=30, xtickstep=5,
                                                                                           ytickmin=0, ytickmax=100,
                                                                                           ytickstep=20, ))

        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                # loss = loss_function(outputs, test_labels)
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                           epochs)
                val_bar.set_postfix(acc=acc/val_num)
        viz.line([acc/val_num], [epoch+1], win='val_acc', update='append',opts=dict(title='val_acc',legend=['acc'],
                                                                                  xtickmin=0,xtickmax=30, xtickstep=5,
                                                                                           ytickmin=0, ytickmax=100,
                                                                                           ytickstep=20, ))
        # viz.line([[test_loss]], [global_step], win='test_loss', update='append', opts=dict(title='test loss',
        #                                                                                    legend=['loss'], xtickmin=0,
        #                                                                                    xtickmax=30, xtickstep=5,
        #                                                                                    ytickmin=0, ytickmax=100,
        #                                                                                    ytickstep=20, ))
        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

    print('Finished Training')


if __name__ == '__main__':
    import argparse

    parser=argparse.ArgumentParser(description='resnet')
    parser.add_argument('--batchsize',type=int,default=8,help='bs')
    parser.add_argument('--epochs', type=int, default=10, help='epochs')
    parser.add_argument('--lr', type=float, default=2e-3, help='lr')
    parser.add_argument('--path', type=str, default='D:/code/pytorchclass/wz_pytorch/', help='path')
    args=parser.parse_args()
    #print(args.batchsize,args.epochs,args.lr,args.path)
    main(bs=args.batchsize,epochs=args.epochs,lr=args.lr,path=args.path)

