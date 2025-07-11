import torch
import torch.nn as nn
import torchvision

from torch.autograd import Variable
import torchvision.models as models

class ResNet18_face_LSTM(nn.Module):

    def __init__(self, pretrained=True, bidirectional=True):
        super(ResNet18_face_LSTM, self).__init__()
        resnet = models.resnet18(pretrained=pretrained)

        self.num_ftrs = resnet.fc.in_features

        self.features = nn.Sequential(*list(resnet.children())[:-2])

        self.avgpool8 = nn.AdaptiveAvgPool2d((1, 1))

        bi_weight=2 if bidirectional else 1
        self.lstm = nn.LSTM(
            input_size=self.num_ftrs,
            hidden_size=int(self.num_ftrs/bi_weight),
            num_layers=3, #3 
            batch_first=True,
            dropout=0.1,
            bias=False,
            bidirectional=bidirectional
        )

        #  binary CE
        self.classifier = nn.Linear(self.num_ftrs, 2)

    def forward(self, x):
        B, C, T, H, W = x.shape

        fs = Variable(torch.zeros(B, T, self.num_ftrs)).to(x.device)

        for ii in range(T):
            h = self.features(x[:, :, ii, :, :])  # [B, C_feat, H_out, W_out]
            embedding = self.avgpool8(h).squeeze(-1).squeeze(-1)  # [B, num_ftrs]
            fs[:, ii, :] = embedding  # fill temporal slot

        out, (hn, cn) = self.lstm(fs)  # out: [B, T, hidden], hn: [num_layers, B, hidden]

        # use the last time step's output
        logits = self.classifier(out[:, -1, :])  # or: self.fc(hn[-1])
        # logits = self.classifier(out.mean(dim=1)) 

        return logits, out[:, -1, :].unsqueeze(1) #.mean(dim=1).unsqueeze(1)

class AU_GAZE_Affect7_LSTM_MLP(nn.Module):

    def __init__(self, bidirectional=True):
        super(AU_GAZE_Affect7_LSTM_MLP, self).__init__()

        # 使用LSTM
        bi_weight=2 if bidirectional else 1
        # self.lstm = nn.LSTM(
        #     input_size=50,  # 与输入特征数量一致 35 AU + 8 Gaze + 5 Expression + 2 Valence/Arousal
        #     hidden_size=int(64/bi_weight),  # 可以调整
        #     num_layers=3,
        #     batch_first=True,
        #     dropout=0.1,
        #     bidirectional=bidirectional # False
        # )
        per_hidden_size = int(64 / (4 * bi_weight))

        # 定义 4 个独立的 LSTM，分别对应 35 AU、8 Gaze、5 Expression、2 Valence/Arousal
        self.lstm_AU = nn.LSTM(
            input_size=35,
            hidden_size=per_hidden_size,
            num_layers=3,
            batch_first=True,
            dropout=0.1,
            bidirectional=bidirectional
        )
        self.lstm_gaze = nn.LSTM(
            input_size=8,
            hidden_size=per_hidden_size,
            num_layers=3,
            batch_first=True,
            dropout=0.1,
            bidirectional=bidirectional
        )
        self.lstm_expr = nn.LSTM(
            input_size=5,
            hidden_size=per_hidden_size,
            num_layers=3,
            batch_first=True,
            dropout=0.1,
            bidirectional=bidirectional
        )
        self.lstm_va = nn.LSTM(
            input_size=2,
            hidden_size=per_hidden_size,
            num_layers=3,
            batch_first=True,
            dropout=0.1,
            bidirectional=bidirectional
        )

        # 拼接后一共是 4 * (per_hidden_size * bi_weight) 维，这里等于 64
        mlp_input_dim = 4 * (per_hidden_size * bi_weight)  # 应该等于 64

        #  binary CE
        self.MLP2_classifier = nn.Sequential(
            nn.Linear(mlp_input_dim, 32),
            nn.LayerNorm(32, eps=1e-6),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(32, 16),
            nn.LayerNorm(16, eps=1e-6),
            nn.ReLU(),
            nn.Linear(16, 2),
        )

    def forward(self, x):
        B, C, T = x.shape  # e.g., [B, 50, 64]
        x = x.permute(0, 2, 1)  # -> [B, T, C=50]

        # 按 feature 维度切分成四段
        # x[:, :, :35] 对应 AU
        # x[:, :, 35:43] 对应 Gaze
        # x[:, :, 43:48] 对应 Expression
        # x[:, :, 48:50] 对应 Valence/Arousal
        x_AU   = x[:, :, 0:35]    # [B, T, 35]
        x_gaze = x[:, :, 35:43]   # [B, T, 8]
        x_expr = x[:, :, 43:48]   # [B, T, 5]
        x_va   = x[:, :, 48:50]   # [B, T, 2]

        # 输出 lstm_out_AU 形状: [B, T, per_hidden_size * bi_weight]
        lstm_out_AU, (_, _)   = self.lstm_AU(x_AU)
        lstm_out_gaze, (_, _) = self.lstm_gaze(x_gaze)
        lstm_out_expr, (_, _) = self.lstm_expr(x_expr)
        lstm_out_va, (_, _)   = self.lstm_va(x_va)

        # 取最后一个时间步的隐藏输出
        # lstm_out_* 的维度都是 [B, T, per_hidden_size*bi_weight]，取 [:, -1, :] 得到 [B, per_hidden_size*bi_weight]
        h_AU_last   = lstm_out_AU[:, -1, :]    # [B, per_hidden_size*bi_weight]
        h_gaze_last = lstm_out_gaze[:, -1, :]  # [B, per_hidden_size*bi_weight]
        h_expr_last = lstm_out_expr[:, -1, :]  # [B, per_hidden_size*bi_weight]
        h_va_last   = lstm_out_va[:, -1, :]    # [B, per_hidden_size*bi_weight]
        
        # 四个特征拼接起来 -> [B, 4 * per_hidden_size*bi_weight]，即 [B, 64]
        h_concat = torch.cat([h_AU_last, h_gaze_last, h_expr_last, h_va_last], dim=1)  # [B, 64]

        logits = self.MLP2_classifier(h_concat)  # -> [B, 2]
        # logits = self.MLP2_classifier(lstm_out.mean(dim=1))  # -> [B, 2]

        return logits, h_concat.unsqueeze(1) #.mean(dim=1).unsqueeze(1)

class AU_GAZE_Affect7_DIVIDE_MLP_MLP(nn.Module):

    def __init__(self):
        super(AU_GAZE_Affect7_DIVIDE_MLP_MLP, self).__init__()

        # 使用LSTM
        per_hidden_size = int(64 / 4)

        self.mlp_AU = nn.Sequential(
            nn.Linear(35, 64),
            nn.LayerNorm(64, eps=1e-6),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(64, 128),
            nn.LayerNorm(128, eps=1e-6),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.mlp_gaze = nn.Sequential(
            nn.Linear(8, 64),
            nn.LayerNorm(64, eps=1e-6),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(64, 128),
            nn.LayerNorm(128, eps=1e-6),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.mlp_expr = nn.Sequential(
            nn.Linear(5, 64),
            nn.LayerNorm(64, eps=1e-6),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(64, 128),
            nn.LayerNorm(128, eps=1e-6),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.mlp_va = nn.Sequential(
            nn.Linear(2, 64),
            nn.LayerNorm(64, eps=1e-6),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(64, 128),
            nn.LayerNorm(128, eps=1e-6),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        # 拼接后一共是 4 * per_hidden_size 维度
        mlp_input_dim = 4 * per_hidden_size  # 应该等于 64

        #  binary CE
        self.MLP2_classifier = nn.Sequential(
            nn.Linear(mlp_input_dim, 32),
            nn.LayerNorm(32, eps=1e-6),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(32, 16),
            nn.LayerNorm(16, eps=1e-6),
            nn.ReLU(),
            nn.Linear(16, 2),
        )

    def forward(self, x):
        B, C, T = x.shape  # e.g., [B, 50, 64]
        x = x.permute(0, 2, 1)  # -> [B, T, C=50]

        # 按 feature 维度切分成四段
        # x[:, :, :35] 对应 AU
        # x[:, :, 35:43] 对应 Gaze
        # x[:, :, 43:48] 对应 Expression
        # x[:, :, 48:50] 对应 Valence/Arousal
        x_AU   = x[:, :, 0:35]    # [B, T, 35]
        x_gaze = x[:, :, 35:43]   # [B, T, 8]
        x_expr = x[:, :, 43:48]   # [B, T, 5]
        x_va   = x[:, :, 48:50]   # [B, T, 2]

        # 输出 mlp_out 形状: [B, T, per_hidden_size * bi_weight]
        h_AU_last = self.mlp_AU(x_AU).mean(dim=1)
        h_gaze_last = self.mlp_gaze(x_gaze).mean(dim=1)
        h_expr_last = self.mlp_expr(x_expr).mean(dim=1)
        h_va_last = self.mlp_va(x_va).mean(dim=1)
        
        # 四个特征拼接起来 -> [B, 4 * per_hidden_size*bi_weight]，即 [B, 64]
        h_concat = torch.cat([h_AU_last, h_gaze_last, h_expr_last, h_va_last], dim=1)  # [B, 64]

        logits = self.MLP2_classifier(h_concat)  # -> [B, 2]
        # logits = self.MLP2_classifier(lstm_out.mean(dim=1))  # -> [B, 2]

        return logits, h_concat.unsqueeze(1) #.mean(dim=1).unsqueeze(1)

class ResNet18_LSTM(nn.Module):

    def __init__(self, pretrained=True, LSTM_layers=1):
        super(ResNet18_LSTM, self).__init__()
        resnet = models.resnet18(pretrained=pretrained)

        self.num_ftrs = resnet.fc.in_features

        self.features = nn.Sequential(*list(resnet.children())[:-2])

        self.rnn = nn.LSTM(input_size=self.num_ftrs, hidden_size=self.num_ftrs, num_layers=LSTM_layers,
                          batch_first=True,
                          bias=False, bidirectional=False)

        #  binary CE
        self.fc = nn.Linear(self.num_ftrs, 2)

        self.avgpool8 = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        [B, C, T, H, W] = x.shape

        fs = Variable(torch.zeros(B, T, self.num_ftrs)).cuda()

        for ii in range(T):
            h = self.features(x[:, :, ii, :, :])
            embedding = self.avgpool8(h).squeeze(-1).squeeze(-1)
            fs[:, ii, :] = embedding

        # pdb.set_trace()

        out, hidden = self.rnn(fs)

        logits = self.fc(out[:, -1, :])

        return logits, out[:, -1, :].unsqueeze(-1)
    
class ResNet18_GRU(nn.Module):

    def __init__(self, pretrained=True, GRU_layers=1):
        super(ResNet18_GRU, self).__init__()
        resnet = models.resnet18(pretrained=pretrained)

        self.num_ftrs = resnet.fc.in_features

        self.features = nn.Sequential(*list(resnet.children())[:-2])

        self.rnn = nn.GRU(input_size=self.num_ftrs, hidden_size=self.num_ftrs, num_layers=GRU_layers,
                          batch_first=True,
                          bias=False, bidirectional=False)

        #  binary CE
        self.fc = nn.Linear(self.num_ftrs, 2)

        self.avgpool8 = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        [B, C, T, H, W] = x.shape

        fs = Variable(torch.zeros(B, T, self.num_ftrs)).cuda()

        for ii in range(T):
            h = self.features(x[:, :, ii, :, :])
            embedding = self.avgpool8(h).squeeze(-1).squeeze(-1)
            fs[:, ii, :] = embedding

        # pdb.set_trace()

        out, hidden = self.rnn(fs)

        logits = self.fc(out[:, -1, :])

        return logits, out[:, -1, :].unsqueeze(-1)
    
class AU_GAZE_Affect7_MLP_MLP(nn.Module):

    def __init__(self, ):
        super(AU_GAZE_Affect7_MLP_MLP, self).__init__()

        #  binary CE
        self.MLP = nn.Sequential(
            nn.Linear(50, 64),
            nn.LayerNorm(64, eps=1e-6),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(64, 128),
            nn.LayerNorm(128, eps=1e-6),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        #  binary CE
        self.MLP2_classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.LayerNorm(32, eps=1e-6),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(32, 16),
            nn.LayerNorm(16, eps=1e-6),
            nn.ReLU(),
            nn.Linear(16, 2),
        )

    def forward(self, x):
        [B, C, T] = x.shape  # [B, 43, T=64]

        x_temp = self.MLP(x.permute(0, 2, 1).reshape(B * T, C))

        h = x_temp.reshape(B, T, 1)

        logits = self.MLP2_classifier(h.squeeze(-1))  # rnn_out [B, 64]

        return logits, h

class conv2d_block(nn.Module):
    def __init__(self, in_channels, out_channels, pad='same', k=3, s=1):
        super(conv2d_block, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=k, padding=pad, stride=s, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        return x


class cnn_face(nn.Module):
    def __init__(self, ):
        super(cnn_face, self).__init__()

        self.conv1 = conv2d_block(3, 64, k=7, pad=(3, 3), s=2)
        self.layer1 = nn.Sequential(
            conv2d_block(64, 64),
            conv2d_block(64, 64),
        )

        self.conv2 = conv2d_block(64, 128, k=3, pad=(1, 1), s=2)
        self.layer2 = nn.Sequential(
            conv2d_block(128, 128),
            conv2d_block(128, 128),
        )

        self.conv3 = conv2d_block(128, 256, k=3, pad=(1, 1), s=2)
        self.layer3 = nn.Sequential(
            conv2d_block(256, 256),
            conv2d_block(256, 256),
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x) + x
        x = self.conv2(x)
        x = self.layer2(x) + x
        x = self.conv3(x)
        x = self.layer3(x) + x

        return self.avg_pool(x)
