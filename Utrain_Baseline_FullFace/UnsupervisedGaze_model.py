import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, input_channel, last = False):
        super(ConvBlock, self).__init__()
        self.Conv_1 = nn.Conv2d(in_channels=input_channel, out_channels=2*input_channel, kernel_size=(3,3), padding='same')
        self.Conv_2 = nn.Conv2d(in_channels=2*input_channel, out_channels=2*input_channel, kernel_size=(3,3), padding='same')
        self.Conv_1l = nn.Conv2d(in_channels=input_channel, out_channels=input_channel, kernel_size=(3,3), padding='same')
        self.Conv_2l = nn.Conv2d(in_channels=input_channel, out_channels=input_channel, kernel_size=(3,3), padding='same')
        self.Conv_skip_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_channel, out_channels=2*input_channel, kernel_size=(1,1)),
            nn.LeakyReLU(),
            nn.BatchNorm2d(2*input_channel)
        )
        self.Conv_skip_1l = nn.Sequential(
            nn.Conv2d(in_channels=input_channel, out_channels=input_channel, kernel_size=(1,1)),
            nn.LeakyReLU(),
            nn.BatchNorm2d(input_channel)
        )
        self.features = nn.Sequential(
            self.Conv_1,
            nn.LeakyReLU(),
            nn.BatchNorm2d(2*input_channel),
            self.Conv_2,
            nn.BatchNorm2d(2*input_channel)
        )
        if last:
            self.features = nn.Sequential(
                self.Conv_1l,
                nn.LeakyReLU(),
                nn.BatchNorm2d(input_channel),
                self.Conv_2l,
                nn.BatchNorm2d(input_channel)
            )
        self.last = last
        self.input_channel = input_channel
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    def forward(self, input):
        if not self.last:
            input_skip = self.Conv_skip_1(input).to(self.device)
        else:
            input_skip = self.Conv_skip_1l(input).to(self.device)
        input = self.features(input)
        input = input_skip + input
        return nn.ReLU()(input)

class DeConvBlock(nn.Module):
    def __init__(self, input_channel, last = False):
        super(DeConvBlock, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=input_channel, out_channels=input_channel//2, kernel_size=(3,3), padding='same'),
            nn.LeakyReLU(),
            nn.BatchNorm2d(input_channel//2),
            nn.Conv2d(in_channels=input_channel//2, out_channels=input_channel//2, kernel_size=(3,3), padding='same'),
            nn.BatchNorm2d(input_channel//2)
        )
        if last: 
            self.features = nn.Sequential(
                nn.Conv2d(in_channels=input_channel, out_channels=input_channel//4, kernel_size=(3,3), padding='same'),
                nn.LeakyReLU(),
                nn.BatchNorm2d(input_channel//4),
                nn.Conv2d(in_channels=input_channel//4, out_channels=input_channel//4, kernel_size=(3,3), padding='same'),
                nn.BatchNorm2d(input_channel//4)
            )
        self.last = last
        self.input_channel = input_channel
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def forward(self, input):
        if not self.last:
            input_skip = nn.Conv2d(in_channels=self.input_channel, out_channels=self.input_channel//2, kernel_size=(1,1))(input).to(self.device)
            input_skip = nn.LeakyReLU()(input_skip)
            input_skip = nn.BatchNorm2d(self.input_channel//2)(input_skip)
        else:
            input_skip = nn.Conv2d(in_channels=self.input_channel, out_channels=self.input_channel//4, kernel_size=(1,1))(input).to(self.device)
            input_skip = nn.LeakyReLU()(input_skip)
            input_skip = nn.BatchNorm2d(self.input_channel//4)(input_skip)
        input = self.features(input)
        input = input_skip + input
        return nn.ReLU()(input)

class GazeRepresentationLearning(nn.Module): #checked
    def __init__(self, channels=64) -> None:
        super(GazeRepresentationLearning, self).__init__()
        self.ConvBlock_1 = ConvBlock(input_channel=channels, last=True)
        self.ConvBlock_2 = ConvBlock(input_channel=channels)
        self.ConvBlock_3 = ConvBlock(input_channel=2*channels, last=True)
        self.ConvBlock_4 = ConvBlock(input_channel=2*channels)
        self.features= nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=channels, kernel_size=(3,3), padding='same'),
            nn.LeakyReLU(),
            nn.BatchNorm2d(channels),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
            self.ConvBlock_1,
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
            self.ConvBlock_2,
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
            self.ConvBlock_3,
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
            self.ConvBlock_4,
            nn.AvgPool2d(kernel_size=(2,3), stride=(2,3)), # 1/3 1/4 for EyeDiap
            nn.Flatten(),
            nn.Linear(in_features=4*channels, out_features=channels//4),
            nn.Linear(in_features=channels//4, out_features=2),
        )
        self.channels = channels
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    def forward(self, input):
        input = self.features(input)
        return input

class GazeRepresentationLearning_KD(nn.Module): #checked
    def __init__(self, channels=64) -> None:
        super(GazeRepresentationLearning_KD, self).__init__()
        self.ConvBlock_1 = ConvBlock(input_channel=channels, last=True)
        self.ConvBlock_2 = ConvBlock(input_channel=channels)
        self.ConvBlock_3 = ConvBlock(input_channel=2*channels, last=True)
        self.ConvBlock_4 = ConvBlock(input_channel=2*channels)

        self.features= nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=channels, kernel_size=(3,3), padding='same'),
            nn.LeakyReLU(),
            nn.BatchNorm2d(channels),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
            self.ConvBlock_1,
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
            self.ConvBlock_2,
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
            self.ConvBlock_3,
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
            self.ConvBlock_4,
            nn.AvgPool2d(kernel_size=(2,3), stride=(2,3)), # 1/3 1/4 for EyeDiap
            nn.Flatten(),
            nn.Linear(in_features=4*channels, out_features=channels//4),
            nn.Linear(in_features=channels//4, out_features=90),
        )
        self.channels = channels

    def forward(self, input):
        input = self.features(input)
        return input


class GazeRepresentationLearning_fullface(nn.Module): #checked
    def __init__(self, channels=64) -> None:
        super(GazeRepresentationLearning_fullface, self).__init__()
        self.ConvBlock_1 = ConvBlock(input_channel=channels, last=True)
        self.ConvBlock_2 = ConvBlock(input_channel=channels)
        self.ConvBlock_3 = ConvBlock(input_channel=2*channels, last=True)
        self.ConvBlock_4 = ConvBlock(input_channel=2*channels)
        self.features= nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=channels, kernel_size=(3,3), padding='same'),
            nn.LeakyReLU(),
            nn.BatchNorm2d(channels),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
            self.ConvBlock_1,
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
            self.ConvBlock_2,
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
            self.ConvBlock_3,
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
            self.ConvBlock_4,
            nn.AvgPool2d(kernel_size=(2,3), stride=(2,3)), # 1/3 1/4 for EyeDiap
            nn.Flatten(),
            nn.Linear(in_features=7168, out_features=channels//4),
            nn.Linear(in_features=channels//4, out_features=2),
        )
        self.channels = channels
    def forward(self, input):
        input = self.features(input)
        return input