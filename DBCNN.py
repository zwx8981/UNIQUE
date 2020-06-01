import torch.nn as nn
from torchvision import models
import torch
import torch.nn.functional as F
from SCNN import SCNN


class DBCNN(nn.Module):

    def __init__(self, config):
        """Declare all needed layers."""
        nn.Module.__init__(self)
        # Convolution and pooling layers of VGG-16.
        self.features1 = models.vgg16(pretrained=True).features
        self.features1 = nn.Sequential(*list(self.features1.children())[:-1])

        scnn = SCNN()
        scnn = torch.nn.DataParallel(scnn).cuda()

        scnn.load_state_dict(torch.load(config.scnn_root))
        self.features2 = scnn.module.features

        # Linear classifier.

        if config.std_modeling:
            outdim = 2
        else:
            outdim = 1

        self.fc = nn.Linear(512 * 128, outdim)

        if config.fc:
            # Freeze all previous layers.
            for param in self.features1.parameters():
                param.requires_grad = False
            for param in self.features2.parameters():
                param.requires_grad = False
            # Initialize the fc layers.
            nn.init.kaiming_normal_(self.fc.weight.data)
            if self.fc.bias is not None:
                nn.init.constant_(self.fc.bias.data, val=0)

    def forward(self, X):
        """Forward pass of the network.
        """
        N = X.size()[0]
        X1 = self.features1(X)
        H = X1.size()[2]
        W = X1.size()[3]
        assert X1.size()[1] == 512
        X2 = self.features2(X)
        H2 = X2.size()[2]
        W2 = X2.size()[3]
        assert X2.size()[1] == 128

        if (H != H2) | (W != W2):
            X2 = F.upsample_bilinear(X2, (H, W))

        X1 = X1.view(N, 512, H * W)
        X2 = X2.view(N, 128, H * W)
        X = torch.bmm(X1, torch.transpose(X2, 1, 2)) / (H * W)  # Bilinear
        assert X.size() == (N, 512, 128)
        X = X.view(N, 512 * 128)
        X = torch.sqrt(X + 1e-8)
        X = torch.nn.functional.normalize(X)
        X = self.fc(X)
        assert X.size() == (N, 2)

        if self.config.std_modeling:
            mean = X [:, 0]
            t = X [:, 1]
            var = nn.functional.softplus(t)
            return mean, var
        else:
            return X
