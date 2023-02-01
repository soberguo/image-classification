import torch
import torchvision.transforms as transforms
from torchvision import datasets
import os
import json
import matplotlib.pyplot as plt
import numpy as np
import torchvision
from vgg import vgg
import torch.nn as nn
from torch import optim
import time




device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)


data_transform={
    'train':transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ]),
    'val':transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
}

#设置数据集路径
data_root=os.path.abspath(os.path.join(os.getcwd(),'./..'))#get data root path\
#print(data_root)
image_path=data_root+'/data_set/flower_data/'


train_dataset=datasets.ImageFolder(root=image_path+'train',transform=data_transform['train'])
val_dataset=datasets.ImageFolder(root=image_path+'val',transform=data_transform['val'])
train_num=len(train_dataset)
#print(train_num)
flower_list=train_dataset.class_to_idx#{'daisy': 0, 'dandelion': 1, 'roses': 2, 'sunflowers': 3, 'tulips': 4}
#print(flower_list)
cla_dict=dict((val,key) for key,val in flower_list.items())#{0: 'daisy', 1: 'dandelion', 2: 'roses', 3: 'sunflowers', 4: 'tulips'}
#print(cla_dict)
json_str=json.dumps(cla_dict,indent=4)
with open('class_indices.json','w') as f:
    f.write(json_str)


batchsize=4
train_loader=torch.utils.data.DataLoader(train_dataset,batch_size=batchsize,shuffle=True,num_workers=0)
val_loader=torch.utils.data.DataLoader(val_dataset,batch_size=batchsize,shuffle=False,num_workers=0)
val_num=len(val_dataset)

#访问数据集的图像
# test_data_iter=iter(val_loader)
# test_image,test_label=test_data_iter.next()
#
# def imshow(img):
#     img=img/2+0.5
#     npimg=img.numpy()
#     plt.imshow(np.transpose(npimg,(1,2,0)))
#     plt.show()
#
# print('  '.join('%5s'% cla_dict[test_label[j].item()] for j in range(4)))
# imshow(torchvision.utils.make_grid(test_image))


net=vgg(model_name='vgg16',num_classes=5,init_weights=True).to(device)
loss_function=nn.CrossEntropyLoss().to(device)
optimizer=optim.Adam(net.parameters(),lr=0.0002)


save_path='./vgg.pth'
best_acc=0.0
for epoch in range(10):
    #train
    net.train()
    running_loss=0.0
    t1=time.perf_counter()
    for step,data in enumerate(train_loader):
        images,labels=data
        images=images.to(device)
        labels=labels.to(device)
        optimizer.zero_grad()
        output=net(images)

        loss=loss_function(output,labels)
        loss.backward()
        optimizer.step()


        running_loss+=loss.item()
        rate=(step+1)/len(train_loader)
        a='*'*int(rate*50)
        b='.'*int((1-rate)*50)
        print('\rtrain loss: {:^3.0f}%[{}->{}]{:.3f}'.format(int(rate*100),a,b,loss),end='')
    print()
    print(time.perf_counter()-t1)


    #val
    net.eval()
    acc=0.0
    with torch.no_grad():
        for data_test in val_loader:
            test_images,test_labels=data_test
            test_images, test_labels =test_images.to(device),test_labels.to(device)
            outputs=net(test_images)
            predict_y=torch.max(outputs,dim=1)[1]
            acc+=(predict_y==test_labels).sum().item()
        acc_test=acc/val_num
        if acc_test>best_acc:
            best_acc=acc_test
            torch.save(net.state_dict(),save_path)
        print('[epoch %d] train loss:%.3f test accuracy:%.3f'%(epoch+1,running_loss/step,acc/val_num))


print('finish train')



