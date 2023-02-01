import torch
import torch.nn as nn

class VGG(nn.Module):
    def __init__(self,features,num_classes=1000,init_weights=False):
        super(VGG,self).__init__()
        self.feature=features
        self.classifier=nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )
        if init_weights:
            self._initialize_weights()
    def forward(self,x):
        x=self.feature(x)
        x=torch.flatten(x,start_dim=1)
        x=self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                # nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_features(cfgs:list):
    layers=[]
    in_channels=3
    for v in cfgs:
        if v=='M':
            layers+=[nn.MaxPool2d(2,2)]
        else:
            conv2d=nn.Conv2d(in_channels,v,kernel_size=3,stride=1,padding=1)
            layers+=[conv2d,nn.ReLU(inplace=True)]
            in_channels=v
    return nn.Sequential(*layers)


cfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def vgg(model_name='vgg16',**kwargs):
    try:
        cfg=cfgs[model_name]
    except:
        print('Warning: model number {} not in cfgs dict!'.format(model_name))
    model=VGG(make_features(cfg),**kwargs)
    return model

def main():
    net=vgg(model_name='vgg16',num_classes=10)
    input=torch.randn(8,3,224,224)
    output=net(input)
    print(output.shape)

if __name__=='__main__':
    main()






