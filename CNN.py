import torch
class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv = torch.nn.Sequential(

            #1.卷积层

            torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2),

            #2.归一化层

            torch.nn.BatchNorm2d(32),

            #3.激活层

            torch.nn.ReLU(),

            #4.池化层

            torch.nn.MaxPool2d(2),

        );

        ###全链接

        self.fc = (torch.nn.
                   Linear(in_features=14*14*32, out_features=10));

    def forward(self, x):
        out = self.conv(x)
        #展开成一维
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
