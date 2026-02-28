# BlazeDOC - Lightweight document corner detector inspired by MediaPipe BlazeFace

class BlazeBlockSingle(nn.Module):
    """
    DepthwiseConv2d (5x5) -> BN -> ReLU -> Conv2d (1x1) -> BN -> Add -> ReLU
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.dw = nn.Conv2d(in_channels, in_channels, kernel_size=5,
                            stride=stride, padding=2, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.pw = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = F.relu(self.bn1(self.dw(x)))
        out = self.bn2(self.pw(out))
        return F.relu(out + residual)


class BlazeDocDetector(nn.Module):
    """
    Input: 416x416x3
      -> Backbone (BlazeFace-style, 5 stages)  -> 13x13x96
      -> Upsampling Neck (ConvTranspose2d x3)   -> 104x104x32
      -> Heatmap Head (Conv + Sigmoid)           -> 104x104x4

    4 corner heatmaps: TL, TR, BR, BL
    270K parameters, ~1.4 MB ONNX
    """
    def __init__(self):
        super().__init__()
        self.backbone = BlazeDocBackbone()   # BlazeBlocks: 24->24->48->48->96->96
        self.neck = UpsamplingNeck()         # ConvTranspose2d: 96->64->48->32
        self.head = HeatmapHead()            # Conv 3x3->1x1->Sigmoid: 32->16->4

    def forward(self, x):
        x = self.backbone(x)  # 416 -> 13
        x = self.neck(x)      # 13  -> 104
        x = self.head(x)      # 104 -> 104 (4 heatmaps)
        return x
