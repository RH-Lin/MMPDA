import torch
import torch.nn as nn
import torchaudio
import torchvision.models as models


class ResNet18_audio_LSTM(nn.Module):

    def __init__(self, pretrained=True, bidirectional=True):
        super(ResNet18_audio_LSTM, self).__init__()
        resnet = models.resnet18(pretrained=pretrained)

        num_ftrs = resnet.fc.in_features

        self.features = nn.Sequential(*list(resnet.children())[:-2])

        # self.avgpool8 = nn.AdaptiveAvgPool2d((1, 1))
        # 使用LSTM
        bi_weight=2 if bidirectional else 1
        self.lstm = nn.LSTM(
            input_size=num_ftrs,  # 通道数作为LSTM输入特征维度
            hidden_size=int(num_ftrs/bi_weight),      # 可选隐藏维度
            num_layers=3,
            batch_first=True,
            dropout=0.1,
            bidirectional=bidirectional # False
        )

        #  binary CE
        self.classifier = nn.Linear(num_ftrs, 2)

        self.avgpool8 = nn.AdaptiveAvgPool2d((1, 7)) # H*W=7*7

    def forward(self, x):
        h = self.features(x) # [B, C, H, W]

        B, C, H, W = h.shape

        # 展平成序列： [B, C, H, W] -> [B, C, W] -> [B, W, C] 
        h_seq = self.avgpool8(h)
        h_seq = h_seq.squeeze(-2).permute(0, 2, 1)
        # 输入到LSTM
        lstm_out, (hn, cn) = self.lstm(h_seq)  # lstm_out: [B, T, 512]
        # h_map = lstm_out[:, -1, :]  # [B, 128] # 可以使用最后一个时间步

        # logits = self.classifier(h_map.squeeze(-1).squeeze(-1))
        # logits = self.classifier(h_map)
        logits = self.classifier(lstm_out[:, -1, :])
        # logits = self.classifier(lstm_out.mean(dim=1))

        # return logits, h_map.unsqueeze(-1).unsqueeze(-1)
        return logits, lstm_out[:, -1, :].unsqueeze(1) #.mean(dim=1).unsqueeze(1)
    
class ResNet18_audio(nn.Module):

    def __init__(self, pretrained=True):
        super(ResNet18_audio, self).__init__()
        resnet = models.resnet18(pretrained=pretrained)

        num_ftrs = resnet.fc.in_features

        self.features = nn.Sequential(*list(resnet.children())[:-2])

        #  binary CE
        self.fc = nn.Linear(num_ftrs, 2)

        self.avgpool8 = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        h = self.features(x)

        regmap8 = self.avgpool8(h)

        logits = self.fc(regmap8.squeeze(-1).squeeze(-1))

        # return logits, regmap8
        return logits, regmap8.squeeze(-1)
    
class ResNet152_audio(nn.Module):

    def __init__(self, pretrained=True):
        super(ResNet152_audio, self).__init__()
        resnet = models.resnet152(pretrained=pretrained)

        num_ftrs = resnet.fc.in_features

        self.features = nn.Sequential(*list(resnet.children())[:-2])

        #  binary CE
        self.fc = nn.Linear(num_ftrs, 2)

        self.avgpool8 = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        h = self.features(x)

        regmap8 = self.avgpool8(h)

        logits = self.fc(regmap8.squeeze(-1).squeeze(-1))

        # return logits, regmap8
        return logits, regmap8