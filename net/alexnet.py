import torch
import torchvision.transforms as transforms
import torch.nn as nn



class AlexNet(nn.Module):
    def __init__(self,num_classes=1000,init_weight=False):
        super(AlexNet,self).__init__()
        self.features=nn.Sequential(
            nn.Conv2d(3,48,kernel_size=11,stride=4,padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2),
            nn.Conv2d(48, 128, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(128, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )


        self.classifier=nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(128*6*6,2048),
            nn.Dropout(p=0.5),
            nn.Linear(2048,2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048,num_classes)
        )

        if init_weight:
            self._initialize_weights()



    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)
            elif isinstance(m,nn.Linear):
                nn.init.normal_(m.weight,0,0.01)
                nn.init.constant_(m.bias,0)

    def forward(self,x):
        x=self.features(x)
        #print(x.shape)
        x=torch.flatten(x,start_dim=1)
        #print(x.shape)
        x=self.classifier(x)
        return x

def main():
    net=AlexNet(num_classes=5,init_weight=False)
    input=torch.randn(32,3,224,224)
    output=net(input)
    print(output.shape)

if __name__=='__main__':
    main()
