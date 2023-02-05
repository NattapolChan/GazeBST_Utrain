import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, input_channel, last = False):
        super(ConvBlock, self).__init__()
        modules = [
            nn.Conv2d(in_channels=input_channel, out_channels=2*input_channel, kernel_size=(3,3), padding='same'),
            nn.LeakyReLU(),
            nn.BatchNorm2d(2*input_channel),
            nn.Conv2d(in_channels=2*input_channel, out_channels=2*input_channel, kernel_size=(3,3), padding='same'),
            nn.BatchNorm2d(2*input_channel)
        ]
        if last:
            modules = [
                nn.Conv2d(in_channels=input_channel, out_channels=input_channel, kernel_size=(3,3), padding='same'),
                nn.LeakyReLU(),
                nn.BatchNorm2d(input_channel),
                nn.Conv2d(in_channels=input_channel, out_channels=input_channel, kernel_size=(3,3), padding='same'),
                nn.BatchNorm2d(input_channel)
            ]
        self.features = nn.Sequential(*modules)
        self.last = last
        self.input_channel = input_channel
    def forward(self, input):
        if not self.last:
            input_skip = nn.Conv2d(in_channels=self.input_channel, out_channels=2*self.input_channel, kernel_size=(1,1))(input)
            input_skip = nn.LeakyReLU()(input_skip)
            input_skip = nn.BatchNorm2d(2*self.input_channel)(input_skip)
        else:
            input_skip = nn.Conv2d(in_channels=self.input_channel, out_channels=self.input_channel, kernel_size=(1,1))(input)
            input_skip = nn.LeakyReLU()(input_skip)
            input_skip = nn.BatchNorm2d(self.input_channel)(input_skip)
        input = self.features(input)
        input = input_skip + input
        return nn.ReLU()(input)

class DeConvBlock(nn.Module):
    def __init__(self, input_channel, last = False):
        super(DeConvBlock, self).__init__()
        modules = [
            nn.Conv2d(in_channels=input_channel, out_channels=input_channel//2, kernel_size=(3,3), padding='same'),
            nn.LeakyReLU(),
            nn.BatchNorm2d(input_channel//2),
            nn.Conv2d(in_channels=input_channel//2, out_channels=input_channel//2, kernel_size=(3,3), padding='same'),
            nn.BatchNorm2d(input_channel//2)
        ]
        if last: 
            modules = [
                nn.Conv2d(in_channels=input_channel, out_channels=input_channel//4, kernel_size=(3,3), padding='same'),
                nn.LeakyReLU(),
                nn.BatchNorm2d(input_channel//4),
                nn.Conv2d(in_channels=input_channel//4, out_channels=input_channel//4, kernel_size=(3,3), padding='same'),
                nn.BatchNorm2d(input_channel//4)
            ]
        self.features = nn.Sequential(*modules)
        self.last = last
        self.input_channel = input_channel

    def forward(self, input):
        if not self.last:
            input_skip = nn.Conv2d(in_channels=self.input_channel, out_channels=self.input_channel//2, kernel_size=(1,1))(input)
            input_skip = nn.LeakyReLU()(input_skip)
            input_skip = nn.BatchNorm2d(self.input_channel//2)(input_skip)
        else:
            input_skip = nn.Conv2d(in_channels=self.input_channel, out_channels=self.input_channel//4, kernel_size=(1,1))(input)
            input_skip = nn.LeakyReLU()(input_skip)
            input_skip = nn.BatchNorm2d(self.input_channel//4)(input_skip)
        input = self.features(input)
        input = input_skip + input
        return nn.ReLU()(input)

class GlobalAlignmentNetwork(nn.Module):
    def __init__(self, channels=16) -> None:
        super(GlobalAlignmentNetwork, self).__init__()
        encoder_shared = [
            nn.Conv2d(in_channels=3, out_channels=channels, kernel_size=(3,3), padding='same'),
            nn.LeakyReLU(),
            nn.BatchNorm2d(channels),
            ConvBlock(input_channel=channels),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
            ConvBlock(input_channel=2*channels),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
        ]
        self.encoder_shared = nn.Sequential(*encoder_shared)
        encoder_main = [
            nn.Conv2d(in_channels=channels*8, out_channels=channels*8, kernel_size=(3,3), padding='same'),
            nn.LeakyReLU(),
            nn.BatchNorm2d(8*channels),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
            ConvBlock(input_channel=channels*8),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
            ConvBlock(input_channel=channels*16, last=True),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)), # avgpool 1/3, 1/4 for Eyediap
            nn.Flatten(),
            nn.Linear(in_features=16*channels, out_features=3),
            nn.Tanh(),
        ]
        self.encoder_main = nn.Sequential(*encoder_main)
        self.channels = channels
    
    def affine_transformation(self, img, theta, height=36, width=60):
        batch = theta.size()[0]
        theta = torch.flatten(theta, start_dim=1)
        # print(f"theta transformation matrix : {theta}")
        coef = theta[:,0]
        iden = torch.eye(2).repeat(batch, 1).view(batch, 2, 2)
        scaling_factor = iden * coef.view(-1,1,1)
        translate = theta[:,1:]
        translate = translate.view(-1,2,1)
        theta = torch.cat([scaling_factor, translate], dim = 2)
        grid = F.affine_grid(theta, torch.Size((batch, 1, height, width)))

        grid = grid.type(torch.float32)
        img = img.type(torch.float32)
        roi = F.grid_sample(img, grid)
        return roi

    def forward(self, input_i, input_o):
        input_i_e = self.encoder_shared(input_i)
        input_o_e = self.encoder_shared(input_o)

        input = torch.cat((input_i_e, input_o_e), dim=1)
        input = self.encoder_main(input)
        
        input_i_t = self.affine_transformation(input_i, input)
        return input_i_t

class GazeRedirectionNetwork(nn.Module):
    def __init__(self, channels=64, dim=(36,60)):
        super(GazeRedirectionNetwork, self).__init__()
        self.convhead = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=channels, kernel_size=(3,3), padding='same'),
            nn.LeakyReLU(),
            nn.BatchNorm2d(channels),
        )
        self.encoder = nn.Sequential(
            ConvBlock(input_channel=channels),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
            ConvBlock(input_channel=2*channels),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
            ConvBlock(input_channel=4*channels, last=True),
            # nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)), #for Eyediap
        )
        self.decoder = nn.Sequential(
            DeConvBlock(input_channel=4*channels),
            nn.UpsamplingBilinear2d(scale_factor=2),
            DeConvBlock(input_channel=2*channels),
            nn.UpsamplingBilinear2d(scale_factor=2),
            DeConvBlock(input_channel=channels, last=True),
            nn.Conv2d(in_channels=channels//4, out_channels=channels//32, kernel_size=(1,1)),
            # nn.Tanh(),
        )
        self.channels = channels
        self.dim = dim

    def forward(self, input_image, input_yaw, input_pitch):
        input_image = self.convhead(input_image)
        input_image = self.encoder(input_image)
        batch_size = input_image.size(0)

        input_yaw = input_yaw.view(batch_size, 1)
        input_pitch = input_pitch.view(batch_size, 1)
        input_yaw = nn.Linear(in_features=1, out_features=9*15)(input_yaw)
        input_pitch = nn.Linear(in_features=1, out_features=9*15)(input_pitch)

        input_yaw = input_yaw.view(batch_size, 1, self.dim[0]//4, self.dim[1]//4)
        input_pitch = input_pitch.view(batch_size, 1, self.dim[0]//4, self.dim[1]//4)

        input_bottleneck = torch.cat((input_image, input_yaw, input_pitch), 1)

        input_deimage = nn.Conv2d(in_channels=4*self.channels+2, out_channels=4*self.channels, kernel_size=(3,3), padding='same')(input_bottleneck)
        input_deimage = nn.LeakyReLU()(input_deimage)
        input_deimage = nn.BatchNorm2d(4*self.channels)(input_deimage)
        input_deimage = self.decoder(input_deimage)
        return input_deimage

class GazeRepresentationLearning(nn.Module): #checked
    def __init__(self, channels=64) -> None:
        super(GazeRepresentationLearning, self).__init__()
        self.features= nn.Sequential(*[
            nn.Conv2d(in_channels=3, out_channels=channels, kernel_size=(3,3), padding='same'),
            nn.LeakyReLU(),
            nn.BatchNorm2d(channels),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
            ConvBlock(input_channel=channels, last=True),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
            ConvBlock(input_channel=channels),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
            ConvBlock(input_channel=2*channels, last=True),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
            ConvBlock(input_channel=2*channels),
            nn.AvgPool2d(kernel_size=(2,3), stride=(2,3)), # 1/3 1/4 for EyeDiap
            nn.Flatten(),
            nn.Linear(in_features=4*channels, out_features=channels//4),
            nn.Linear(in_features=channels//4, out_features=2),
            # nn.Tanh(),
        ])
        self.channels = channels
    def forward(self, input):
        input = self.features(input)
        return input

class UnsupervisedGazeNetwork(nn.Module):
    def __init__(self) -> None:
        super(UnsupervisedGazeNetwork, self).__init__()
        self.shared_gazeEstimation = GazeRepresentationLearning()
        self.gazeRedirection = GazeRedirectionNetwork() #output grid for redirection
        self.align = GlobalAlignmentNetwork()
        # self.VGG = [VGG_3, VGG_8, VGG_13]
        
    def forward(self, input_i, input_o):
        height = input_i.size(2)
        width = input_i.size(3)
        angle_i = self.shared_gazeEstimation(input_i)
        angle_o = self.shared_gazeEstimation(input_o)

        self.input_i_t = self.align(input_i, input_o)
        self.input_i_t = input_o

        angle_dif = angle_i - angle_o
        angle_dif_yaw = angle_dif[:,0]
        angle_dif_pitch = angle_dif[:,1]
        grid_i_t = self.gazeRedirection(self.input_i_t,  angle_dif_yaw, angle_dif_pitch)
        grid_i_t = torch.permute(grid_i_t, (0, 2, 3, 1))

        output = F.grid_sample(self.input_i_t, grid_i_t)

        feature_i, feature_o = [], []
        # for each in self.VGG:
        #     feature_i.append(each(input_o))
        #     feature_o.append(each(output))
        return output, feature_i, feature_o