import torch
import torch.nn as nn
from torchvision.models import resnet50


class Flatten(nn.Module):

    def forward(self, x):
        return x.view(x.size(0), -1)


class FaceNetModel(nn.Module):
    def __init__(self, embedding_size, num_classes, pretrained=False):
        super(FaceNetModel, self).__init__()

        self.model = resnet50(pretrained)
        self.embedding_size = embedding_size
        cnn = nn.Sequential(
            self.model.conv1,
            self.model.bn1,
            self.model.relu,
            self.model.maxpool,
            self.model.layer1,
            self.model.layer2,
            self.model.layer3,
            self.model.layer4)
        self.cnn = torch.nn.DataParallel(cnn)

        fc = nn.Sequential(
            Flatten(),
            nn.Linear(25088, self.embedding_size))
        self.model.fc = torch.nn.DataParallel(fc)

        classifier = nn.Linear(self.embedding_size, num_classes)
        self.model.classifier = torch.nn.DataParallel(classifier)

    def l2_norm(self, input):
        input_size = input.size()
        buffer = torch.pow(input, 2)
        normp = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(normp)
        _output = torch.div(input, norm.view(-1, 1).expand_as(input))
        output = _output.view(input_size)

        return output

    def freeze_all(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze_all(self):
        for param in self.model.parameters():
            param.requires_grad = True

    def freeze_fc(self):
        for param in self.model.fc.parameters():
            param.requires_grad = False

    def unfreeze_fc(self):
        for param in self.model.fc.parameters():
            param.requires_grad = True

    def freeze_classifier(self):
        for param in self.model.classifier.parameters():
            param.requires_grad = False

    def unfreeze_classifier(self):
        for param in self.model.classifier.parameters():
            param.requires_grad = True

    # returns face embedding(embedding_size)
    def forward(self, x):
        x = self.cnn(x)
        x = self.model.fc(x)

        self.features = self.l2_norm(x)
        # Multiply by alpha = 10 as suggested in https://arxiv.org/pdf/1703.09507.pdf
        alpha = 10
        self.features = self.features * alpha
        return self.features

    def forward_classifier(self, x):
        features = self.forward(x)
        res = self.model.classifier(features)
        return res
