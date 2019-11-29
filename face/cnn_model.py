import torch
import torch.nn as nn
import torch.nn.functional as F
#from torchvision import models

class GazeTrain(nn.Module):
    """
    Resnet150の最終層をLSTMに入力して状態を推定するモデル
    """

    def __init__(self, num_layers=1, hidden_size=256):
        super(GazeTrain, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=32,
                               kernel_size=5,
                               stride=1,
                               padding=0)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(32,32,5,1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(32,32,3,1)
        self.conv4 = nn.Conv2d(32,32,3,1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv5 = nn.Conv2d(32,32,3,1)
        self.bn3 = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(46208, 1024)
        #self.fc1 = nn.Linear(6*38*32, 1024)
        self.relu1 = nn.ReLU()
        self.dr1 = nn.Dropout()
        self.fc2 = nn.Linear(1024, 64)
        self.relu2 = nn.ReLU()
        self.dr2 = nn.Dropout()
        self.fc3 = nn.Linear(64, 2)

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, x):
        for f in [self.conv1, self.relu, self.conv2, self.relu, self.bn1, self.pool1, self.conv3, self.relu, self.conv4, self.relu, self.bn2, self.conv5, self.relu, self.bn3]:
            x = f(x)

        x = x.view(len(x),-1)
        x = self.dr1(self.relu1(self.fc1(x)))
        h = self.dr2(self.relu2(self.fc2(x)))
        y = self.fc3(h)
        return y

    def get_middle(self,x):
        for f in [self.conv1, self.relu, self.conv2, self.relu, self.bn1, self.pool1, self.conv3, self.relu, self.conv4, self.relu, self.bn2, self.conv5, self.relu, self.bn3]:
            x = f(x)

        x = x.view(len(x),-1)
        x = self.dr1(self.relu1(self.fc1(x)))
        h = self.dr2(self.relu2(self.fc2(x)))
        return h


