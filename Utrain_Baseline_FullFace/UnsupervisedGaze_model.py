import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

VGG = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
for param in VGG.parameters():
    param.requires_grad = False
VGG_3 = nn.Sequential(*list(VGG.children())[0][:5])
VGG_8 = nn.Sequential(*list(VGG.children())[0][:13])
VGG_13 = nn.Sequential(*list(VGG.children())[0][:22])

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
        self.Conv_skip = nn.Sequential(
            nn.Conv2d(in_channels=self.input_channel, out_channels=self.input_channel//2, kernel_size=(1,1)),
            nn.LeakyReLU(),
            nn.BatchNorm2d(self.input_channel//2),
        )
        self.Conv_skip_l = nn.Sequential(
            nn.Conv2d(in_channels=self.input_channel, out_channels=self.input_channel//4, kernel_size=(1,1)),
            nn.LeakyReLU(),
            nn.BatchNorm2d(self.input_channel//4),
        )

    def forward(self, input):
        if not self.last:
            input_skip = self.Conv_skip(input).to(device)
        else:
            input_skip = self.Conv_skip_l(input).to(device)
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

class GlobalAlignmentNetwork(nn.Module):
    # global alignment network, input: 2 images, output: 1 image
    def __init__(self, channels=16) -> None:
        super(GlobalAlignmentNetwork, self).__init__()
        self.encoder_shared = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=channels, kernel_size=(3,3), padding='same'),
            nn.LeakyReLU(),
            nn.BatchNorm2d(channels),
            ConvBlock(input_channel=channels),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
            ConvBlock(input_channel=2*channels),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
        )
        self.encoder_main = nn.Sequential(
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
        )
        self.channels = channels
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    def affine_transformation(self, img, theta, height=36, width=60):
        batch = theta.size()[0]
        theta = torch.flatten(theta, start_dim=1)
        coef = theta[:,0].to(self.device)
        iden = torch.eye(2).repeat(batch, 1).view(batch, 2, 2).to(self.device)
        scaling_factor = iden * coef.view(-1,1,1).to(self.device)
        translate = theta[:,1:].to(self.device)
        translate = translate.view(-1,2,1)
        theta = torch.cat([scaling_factor, translate], dim = 2).to(self.device)
        grid = F.affine_grid(theta, torch.Size((batch, 1, height, width))).to(self.device)
        grid = grid.type(torch.float32)
        img = img.type(torch.float32)
        roi = F.grid_sample(img, grid).to(self.device)
        return roi

    def forward(self, input_i, input_o):
        input_i_e = self.encoder_shared(input_i).to(self.device)
        input_o_e = self.encoder_shared(input_o).to(self.device)

        input = torch.cat((input_i_e, input_o_e), dim=1).to(self.device)
        input = self.encoder_main(input)
        
        input_i_t = self.affine_transformation(input_i, input)
        return input_i_t

class GazeRedirectionNetwork(nn.Module):
    # gaze redirection network, input: image, 2 2-dim vectors, output: 1 image
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
            nn.Conv2d(in_channels=4*channels+2, out_channels=4*channels, kernel_size=(3,3), padding='same'),
            nn.LeakyReLU(),
            nn.BatchNorm2d(4*channels),
            DeConvBlock(input_channel=4*channels),
            nn.UpsamplingBilinear2d(scale_factor=2),
            DeConvBlock(input_channel=2*channels),
            nn.UpsamplingBilinear2d(scale_factor=2),
            DeConvBlock(input_channel=channels, last=True),
            nn.Conv2d(in_channels=channels//4, out_channels=channels//32, kernel_size=(1,1)),
            nn.Tanh(),
        )
        self.linear_yaw = nn.Linear(in_features=1, out_features=9*15)
        self.linear_pitch = nn.Linear(in_features=1, out_features=9*15)
        self.channels = channels
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dim = dim

    def forward(self, input_image, input_yaw, input_pitch):
        input_image = self.convhead(input_image)
        input_image = self.encoder(input_image)
        batch_size = input_image.size(0)

        input_yaw = input_yaw.view(batch_size, 1).to(self.device)
        input_pitch = input_pitch.view(batch_size, 1).to(self.device)
        input_yaw = self.linear_yaw(input_yaw)
        input_pitch = self.linear_pitch(input_pitch)

        input_yaw = input_yaw.view(batch_size, 1, self.dim[0]//4, self.dim[1]//4).to(self.device)
        input_pitch = input_pitch.view(batch_size, 1, self.dim[0]//4, self.dim[1]//4).to(self.device)

        input_bottleneck = torch.cat((input_image, input_yaw, input_pitch), 1).to(self.device)
        input_deimage = self.decoder(input_bottleneck).to(self.device)
        return input_deimage

class UnsupervisedGazeNetwork(nn.Module):
    # Entire unsupervised architecture 
    # (   
    #     combination of 
    #     1: 2 shared gaze representation learnings, input: 1 image, output 2-dim vector (yaw and pitch)
    #     2: global alignment network, input: 2 images, output: 1 image
    #     3: gaze redirection network, input: image, 2 2-dim vectors, output: 1 image
    # )

    def __init__(self, height=36, width=60) -> None:
        super(UnsupervisedGazeNetwork, self).__init__()
        self.shared_gazeEstimation = GazeRepresentationLearning()
        self.gazeRedirection = GazeRedirectionNetwork(dim=(height, width))
        self.align = GlobalAlignmentNetwork()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.VGG = [VGG_3.to(self.device), VGG_8.to(self.device), VGG_13.to(self.device)]
        
    def forward(self, input_i, input_o, epoch):
        angle_i = self.shared_gazeEstimation(input_i).to(self.device)
        angle_o = self.shared_gazeEstimation(input_o).to(self.device)

        if epoch <= 2:
            angle_i, angle_o = nn.Tanh()(angle_i), nn.Tanh()(angle_o)

        input_i_t = self.align(input_i, input_o).to(self.device)

        angle_dif = (angle_i - angle_o).to(self.device)
        angle_dif_yaw = angle_dif[:,0].to(self.device)
        angle_dif_pitch = angle_dif[:,1].to(self.device)
        grid_i_t = self.gazeRedirection(input_i_t,  angle_dif_yaw, angle_dif_pitch).to(self.device)
        grid_i_t = torch.permute(grid_i_t, (0, 2, 3, 1))
        output = F.grid_sample(input_i_t, grid_i_t).to(self.device)
        feature_i, feature_o = [], []
        for each in self.VGG:
            feature_i.append(each(input_o))
            feature_o.append(each(output))
        return output, feature_i, feature_o

class Loss(nn.Module):
    def __init__(self, height=36, width=60, coef_pixel=1.0, coef_perceptual=0.02, coef_spatial=0.1, coef_warp_reg=0.25) -> None: 
        super(Loss, self).__init__()
        self.height= height
        self.width = width
        self.coef_pixel = coef_pixel
        self.coef_perceptual = coef_perceptual
        self.coef_spatial = coef_spatial
        self.loss__style = 0
        self.loss__feature = 0
        self.loss__pixel = 0

    def pixel_loss(self, image_i_t, image_o):
        discrepancy = torch.abs(image_i_t - image_o)
        self.loss__pixel = 1/torch.numel(image_i_t) * torch.sum(discrepancy)
        return self.loss__pixel
    
    def feature_loss(self, feature_i, feature_o):
        assert len(feature_i) == len(feature_o), f'uncompatible feature size: {len(feature_i)} and {len(feature_o)}'
        assert feature_i[0].size() == feature_o[0].size(), "intermediate feature got different size"
        self.loss__feature = 0
        for i in range(len(feature_i)):
          self.loss__feature += torch.linalg.norm(feature_i[i]-feature_o[i]) / torch.numel(feature_i[i])
        return self.loss__feature

    def style_loss(self, feature_i, feature_o):
        assert len(feature_i) == len(feature_o), f'uncompatible feature size: {len(feature_i)} and {len(feature_o)}'
        assert feature_i[0].size() == feature_o[0].size(), "intermediate feature got different size"
        accumulate = 0
        for each_layer in range(len(feature_i)):
            channel_size = feature_o[each_layer].size(1)
            spatial_size = feature_i[each_layer].size(2) * feature_i[each_layer].size(3)
            mat_i = torch.flatten(feature_i[each_layer], start_dim=1)
            mat_o = torch.flatten(feature_o[each_layer], start_dim=1)
            gram_i = 1/spatial_size * torch.matmul(mat_i, torch.transpose(mat_i, 0, 1))
            gram_o = 1/spatial_size * torch.matmul(mat_o, torch.transpose(mat_o, 0, 1))

            accumulate += 1/channel_size * 1/channel_size * torch.linalg.norm(gram_i-gram_o)
        return accumulate
        
    def forward(self, image_i_t, image_o, feature_i, feature_o):
        # 
        return  self.coef_pixel*self.pixel_loss(image_i_t, image_o) + self.coef_perceptual*self.feature_loss(feature_i, feature_o) + self.coef_spatial * self.style_loss(feature_i, feature_o)
