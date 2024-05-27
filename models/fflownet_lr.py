import torch
import torch.nn as nn
import torch.nn.functional as F
from spatial_correlation_sampler import SpatialCorrelationSampler

__all__ = ["FastFlowNet"]

class Correlation(nn.Module):
    def __init__(self, max_displacement):
        super(Correlation, self).__init__()
        self.max_displacement = max_displacement
        self.kernel_size = 2 * max_displacement + 1
        self.corr = SpatialCorrelationSampler(1, self.kernel_size, 1, 0, 1)

    def forward(self, x, y):
        b, c, h, w = x.shape
        return self.corr(x, y).view(b, -1, h, w) / c

def convrelu(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=bias),
        nn.LeakyReLU(0.1, inplace=True)
    )

def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size, stride, padding, bias=True)

class Decoder(nn.Module):
    def __init__(self, in_channels, groups):
        super(Decoder, self).__init__()
        self.in_channels = in_channels
        self.groups = groups
        self.conv1 = convrelu(in_channels, 96, 3, 1)
        self.conv2 = convrelu(96, 96, 3, 1, groups=groups)
        self.conv3 = convrelu(96, 96, 3, 1, groups=groups)
        self.conv4 = convrelu(96, 96, 3, 1, groups=groups)
        self.conv5 = convrelu(96, 64, 3, 1)
        self.conv6 = convrelu(64, 32, 3, 1)
        self.conv7 = nn.Conv2d(32, 2, 3, 1, 1)

    def channel_shuffle(self, x, groups):
        b, c, h, w = x.size()
        channels_per_group = c // groups
        x = x.view(b, groups, channels_per_group, h, w)
        x = x.transpose(1, 2).contiguous()
        x = x.view(b, -1, h, w)
        return x

    def forward(self, x):
        if self.groups == 1:
            out = self.conv7(self.conv6(self.conv5(self.conv4(self.conv3(self.conv2(self.conv1(x)))))))
        else:
            out = self.conv1(x)
            out = self.channel_shuffle(self.conv2(out), self.groups)
            out = self.channel_shuffle(self.conv3(out), self.groups)
            out = self.channel_shuffle(self.conv4(out), self.groups)
            out = self.conv7(self.conv6(self.conv5(out)))
        return out

class FastFlowNet(nn.Module):
    def __init__(self, groups=3):
        super(FastFlowNet, self).__init__()
        self.groups = groups
        self.pconv1_1 = convrelu(3, 16, 3, 2)
        self.pconv1_2 = convrelu(16, 16, 3, 1)
        self.pconv2_1 = convrelu(16, 32, 3, 2)
        self.pconv2_2 = convrelu(32, 32, 3, 1)
        self.pconv2_3 = convrelu(32, 32, 3, 1)
        self.pconv3_1 = convrelu(32, 64, 3, 2)
        self.pconv3_2 = convrelu(64, 64, 3, 1)
        self.pconv3_3 = convrelu(64, 64, 3, 1)

        self.corr = Correlation(4)
        self.index = torch.tensor([0, 2, 4, 6, 8,
                                   10, 12, 14, 16,
                                   18, 20, 21, 22, 23, 24, 26,
                                   28, 29, 30, 31, 32, 33, 34,
                                   36, 38, 39, 40, 41, 42, 44,
                                   46, 47, 48, 49, 50, 51, 52,
                                   54, 56, 57, 58, 59, 60, 62,
                                   64, 66, 68, 70,
                                   72, 74, 76, 78, 80])

        self.rconv2 = convrelu(32, 32, 3, 1)
        self.rconv3 = convrelu(64, 32, 3, 1)
        self.rconv4 = convrelu(64, 32, 3, 1)
        self.rconv5 = convrelu(64, 32, 3, 1)
        self.rconv6 = convrelu(64, 32, 3, 1)

        self.up3 = deconv(2, 2)
        self.up4 = deconv(2, 2)
        self.up5 = deconv(2, 2)
        self.up6 = deconv(2, 2)

        self.decoder2 = Decoder(87, groups)
        self.decoder3 = Decoder(87, groups)
        self.decoder4 = Decoder(87, groups)
        self.decoder5 = Decoder(87, groups)
        self.decoder6 = Decoder(87, groups)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def warp(self, x, flo):
        B, C, H, W = x.size()
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat([xx, yy], 1).to(x)
        vgrid = grid + flo
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0
        vgrid = vgrid.permute(0, 2, 3, 1)
        output = F.grid_sample(x, vgrid, mode='bilinear', align_corners=True)
        return output

    def forward(self, x):
        img1 = F.interpolate(x[:, :3, :, :], scale_factor=2, mode='lanczos')
        img2 = F.interpolate(x[:, 3:6, :, :], scale_factor=2, mode='lanczos')
        f11 = self.pconv1_2(self.pconv1_1(img1))
        f21 = self.pconv1_2(self.pconv1_1(img2))
        f12 = self.pconv2_3(self.pconv2_2(self.pconv2_1(f11)))
        f22 = self.pconv2_3(self.pconv2_2(self.pconv2_1(f21)))
        f13 = self.pconv3_3(self.pconv3_2(self.pconv3_1(f12)))
        f23 = self.pconv3_3(self.pconv3_2(self.pconv3_1(f22)))
        f14 = F.avg_pool2d(f13, kernel_size=(2, 2), stride=(2, 2))
        f24 = F.avg_pool2d(f23, kernel_size=(2, 2), stride=(2, 2))
        f15 = F.avg_pool2d(f14, kernel_size=(2, 2), stride=(2, 2))
        f25 = F.avg_pool2d(f24, kernel_size=(2, 2), stride=(2, 2))
        f16 = F.avg_pool2d(f15, kernel_size=(2, 2), stride=(2, 2))
        f26 = F.avg_pool2d(f25, kernel_size=(2, 2), stride=(2, 2))

        
        flow7_up = estimate_optical_flow(img1, img2).to(f15)
        cv6 = torch.index_select(self.corr(f16, f26), dim=1, index=self.index.to(f16).long())
        r16 = self.rconv6(f16)
        cat6 = torch.cat([cv6, r16, flow7_up], 1)
        flow6 = self.decoder6(cat6)

        flow6_up = self.up6(flow6)
        f25_w = self.warp(f25, flow6_up * 0.625)
        cv5 = torch.index_select(self.corr(f15, f25_w), dim=1, index=self.index.to(f15).long())
        r15 = self.rconv5(f15)
        cat5 = torch.cat([cv5, r15, flow6_up], 1)
        flow5 = self.decoder5(cat5) + flow6_up

        flow5_up = self.up5(flow5)
        f24_w = self.warp(f24, flow5_up * 1.25)
        cv4 = torch.index_select(self.corr(f14, f24_w), dim=1, index=self.index.to(f14).long())
        r14 = self.rconv4(f14)
        cat4 = torch.cat([cv4, r14, flow5_up], 1)
        flow4 = self.decoder4(cat4) + flow5_up

        flow4_up = self.up4(flow4)
        f23_w = self.warp(f23, flow4_up * 2.5)
        cv3 = torch.index_select(self.corr(f13, f23_w), dim=1, index=self.index.to(f13).long())
        r13 = self.rconv3(f13)
        cat3 = torch.cat([cv3, r13, flow4_up], 1)
        flow3 = self.decoder3(cat3) + flow4_up

        flow3_up = self.up3(flow3)
        f22_w = self.warp(f22, flow3_up * 5.0)
        cv2 = torch.index_select(self.corr(f12, f22_w), dim=1, index=self.index.to(f12).long())
        r12 = self.rconv2(f12)
        cat2 = torch.cat([cv2, r12, flow3_up], 1)
        flow2 = self.decoder2(cat2) + flow3_up

        if self.training:
            return flow2, flow3, flow4, flow5, flow6
        else:
            return flow2

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if "weight" in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if "bias" in name]

    def estimate_optical_flow(batch_images1, batch_images2, levels=4, window_size=5, max_iterations=10):
        def preprocess_image(image):
            # Convert to grayscale
            grayscale = 0.299 * image[:, 0, :, :] + 0.587 * image[:, 1, :, :] + 0.114 * image[:, 2, :, :]
            return grayscale.unsqueeze(1)

        def build_pyramid(image, levels):
            pyramid = [image]
            for _ in range(1, levels):
                image = F.interpolate(image, scale_factor=0.5, mode='bilinear', align_corners=False)
                pyramid.append(image)
            return pyramid

        def lucas_kanade_optical_flow(I1, I2, window_size=5, max_iterations=10):
            batch_size, _, h, w = I1.shape
            u = torch.zeros(batch_size, 1, h, w, device=I1.device)
            v = torch.zeros(batch_size, 1, h, w, device=I1.device)

            I1x = F.conv2d(I1, torch.tensor([[[[-1, 1], [-1, 1]]]], device=I1.device), padding=1)
            I1y = F.conv2d(I1, torch.tensor([[[[-1, -1], [1, 1]]]], device=I1.device), padding=1)
            
            I1t = I2 - I1

            for _ in range(max_iterations):
                u_mean = F.avg_pool2d(u, window_size, stride=1, padding=window_size//2)
                v_mean = F.avg_pool2d(v, window_size, stride=1, padding=window_size//2)

                I1x_u = I1x * u_mean
                I1y_v = I1y * v_mean

                b1 = I1t + I1x_u + I1y_v
                A11 = I1x * I1x
                A12 = I1x * I1y
                A22 = I1y * I1y

                detA = A11 * A22 - A12 * A12

                u_new = (A22 * b1 - A12 * b1) / (detA + 1e-8)
                v_new = (A11 * b1 - A12 * b1) / (detA + 1e-8)

                u = u_new
                v = v_new

            return u, v

        def pyramidal_lucas_kanade(I1, I2, levels=4, window_size=5, max_iterations=10):
            pyramid1 = build_pyramid(I1, levels)
            pyramid2 = build_pyramid(I2, levels)
            
            flow_u, flow_v = lucas_kanade_optical_flow(pyramid1[-1], pyramid2[-1], window_size, max_iterations)
            
            for level in range(levels-2, -1, -1):
                flow_u = F.interpolate(flow_u, scale_factor=2, mode='bilinear', align_corners=False) * 2
                flow_v = F.interpolate(flow_v, scale_factor=2, mode='bilinear', align_corners=False) * 2

                u, v = lucas_kanade_optical_flow(pyramid1[level], pyramid2[level], window_size, max_iterations)
                flow_u += u
                flow_v += v
            
            return torch.cat((flow_u, flow_v), dim=1)

        grayscale1 = preprocess_image(batch_images1)
        grayscale2 = preprocess_image(batch_images2)
        
        flow = pyramidal_lucas_kanade(grayscale1, grayscale2, levels, window_size, max_iterations)
        
        return flow


def fflownet_lr(data=None):
    model = FastFlowNet()
    if data is not None:
        model.load_state_dict(data["state_dict"])
    return model
