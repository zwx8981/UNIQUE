import torch.nn as nn


# from torchvision import models
# import torch

def weight_init(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


class SCNN(nn.Module):

    def __init__(self):
        """Declare all needed layers."""
        super(SCNN, self).__init__()

        # Linear classifier.

        self.num_class = 39
        self.features = nn.Sequential(nn.Conv2d(3, 48, 3, 1, 1), nn.ReLU(inplace=True),
                                      nn.Conv2d(48, 48, 3, 2, 1), nn.ReLU(inplace=True),
                                      nn.Conv2d(48, 64, 3, 1, 1), nn.ReLU(inplace=True),
                                      nn.Conv2d(64, 64, 3, 2, 1), nn.ReLU(inplace=True),
                                      nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(inplace=True),
                                      nn.Conv2d(64, 64, 3, 2, 1), nn.ReLU(inplace=True),
                                      nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(inplace=True),
                                      nn.Conv2d(128, 128, 3, 1, 1), nn.ReLU(inplace=True),
                                      nn.Conv2d(128, 128, 3, 2, 1), nn.ReLU(inplace=True))

        weight_init(self.features)
        self.pooling = nn.AvgPool2d(14)
        self.projection = nn.Sequential(nn.Conv2d(128, 256, 1, 1, 0), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
                                        nn.Conv2d(256, 256, 1, 1, 0), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        weight_init(self.projection)
        self.classifier = nn.Linear(256, self.num_class)
        weight_init(self.classifier)

    def forward(self, x):
        N = x.size()[0]
        assert x.size() == (N, 3, 224, 224)
        x = self.features(x)
        assert x.size() == (N, 128, 14, 14)
        x = self.pooling(x)
        assert x.size() == (N, 128, 1, 1)
        x = self.projection(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        assert x.size() == (N, self.num_class)

        return x
