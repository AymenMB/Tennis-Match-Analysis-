# --- START OF FILE tennistrack.py ---

import torch.nn as nn
import torch

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, pad=1, stride=1, bias=True):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=bias),
            nn.ReLU(inplace=True), # Use inplace ReLU for memory efficiency
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.block(x)

class CourtKeypointNet(nn.Module):
    def __init__(self, out_channels=15): # 14 keypoints + 1 center point
        super().__init__()
        self.out_channels = out_channels
        dropout_prob = 0.1 # Define dropout probability

        # Encoder
        self.conv1 = ConvBlock(in_channels=3, out_channels=64)
        self.conv2 = ConvBlock(in_channels=64, out_channels=64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop1 = nn.Dropout2d(p=dropout_prob) # Added Dropout

        self.conv3 = ConvBlock(in_channels=64, out_channels=128)
        self.conv4 = ConvBlock(in_channels=128, out_channels=128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop2 = nn.Dropout2d(p=dropout_prob) # Added Dropout

        self.conv5 = ConvBlock(in_channels=128, out_channels=256)
        self.conv6 = ConvBlock(in_channels=256, out_channels=256)
        self.conv7 = ConvBlock(in_channels=256, out_channels=256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop3 = nn.Dropout2d(p=dropout_prob) # Added Dropout

        # Bottleneck
        self.conv8 = ConvBlock(in_channels=256, out_channels=512)
        self.conv9 = ConvBlock(in_channels=512, out_channels=512)
        self.conv10 = ConvBlock(in_channels=512, out_channels=512)

        # Decoder
        self.ups1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) # Use bilinear upsampling
        # Add concatenation with skip connection features here if adopting a true U-Net
        # For now, following TennisTrack structure: Upsample -> Conv helps merge
        self.conv11 = ConvBlock(in_channels=512, out_channels=256) # Assuming no concat: input is 512
        # If concatenating skip from conv7 (256 channels): in_channels=512+256 = 768
        self.conv12 = ConvBlock(in_channels=256, out_channels=256)
        self.conv13 = ConvBlock(in_channels=256, out_channels=256)

        self.ups2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # Add concatenation with skip connection features here (e.g., from conv4: 128 channels)
        self.conv14 = ConvBlock(in_channels=256, out_channels=128) # Assuming no concat: input is 256
        # If concatenating skip: in_channels=256+128 = 384
        self.conv15 = ConvBlock(in_channels=128, out_channels=128)

        self.ups3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # Add concatenation with skip connection features here (e.g., from conv2: 64 channels)
        self.conv16 = ConvBlock(in_channels=128, out_channels=64) # Assuming no concat: input is 128
        # If concatenating skip: in_channels=128+64 = 192
        self.conv17 = ConvBlock(in_channels=64, out_channels=64)

        # Final Layer
        self.conv18 = nn.Conv2d(in_channels=64, out_channels=self.out_channels, kernel_size=1, padding=0) # 1x1 conv for final output

        self._init_weights()

    def forward(self, x):
        # Encoder
        x1 = self.conv1(x)
        x1 = self.conv2(x1)
        x_pool1 = self.pool1(x1)
        if self.training: x_pool1 = self.drop1(x_pool1) # Apply dropout only during training

        x2 = self.conv3(x_pool1)
        x2 = self.conv4(x2)
        x_pool2 = self.pool2(x2)
        if self.training: x_pool2 = self.drop2(x_pool2)

        x3 = self.conv5(x_pool2)
        x3 = self.conv6(x3)
        x3 = self.conv7(x3)
        x_pool3 = self.pool3(x3)
        if self.training: x_pool3 = self.drop3(x_pool3)

        # Bottleneck
        b = self.conv8(x_pool3)
        b = self.conv9(b)
        b = self.conv10(b)

        # Decoder
        # Note: Add skip connections here (e.g., torch.cat([self.ups1(b), x3], dim=1))
        # and adjust subsequent conv in_channels if changing to a standard U-Net.
        # Keeping TennisTrack structure for now:
        d1 = self.ups1(b)
        d1 = self.conv11(d1)
        d1 = self.conv12(d1)
        d1 = self.conv13(d1)

        d2 = self.ups2(d1)
        d2 = self.conv14(d2)
        d2 = self.conv15(d2)

        d3 = self.ups3(d2)
        d3 = self.conv16(d3)
        d3 = self.conv17(d3)

        out = self.conv18(d3)
        return out

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu') # Kaiming init often better
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear): # Just in case linear layers were added
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = CourtKeypointNet(out_channels=15).to(device) # Ensure correct out_channels
    # Set model to training mode to test dropout path
    model.train()
    # Network expects 640x360 input based on README, adjust if needed
    inp = torch.rand(1, 3, 360, 640).to(device)
    out = model(inp)
    print('out shape = {}'.format(out.shape)) # Output shape should be [1, 15, 360, 640]
