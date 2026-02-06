import torch
import torch.nn as nn


class CBAM(nn.Module):
    """CBAM 代码不变"""

    def __init__(self, c1, kernel_size=7):
        super().__init__()
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c1, c1 // 16, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1 // 16, c1, 1, bias=False),
            nn.Sigmoid()
        )
        self.sa = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x * self.ca(x)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        return x * self.sa(out)


class SEAttention(nn.Module):
    """Squeeze-and-Excitation 模块 - 延迟初始化版 (解决通道不匹配)"""

    def __init__(self, c1=None, reduction=16, *args, **kwargs):
        super().__init__()
        self.reduction = reduction
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 初始时不创建卷积层
        self.fc = None

    def forward(self, x):
        # 第一次运行时，根据输入 x 的通道数动态创建卷积层
        if self.fc is None:
            channel = x.shape[1]
            self.fc = nn.Sequential(
                nn.Conv2d(channel, channel // self.reduction, 1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // self.reduction, channel, 1, bias=False),
                nn.Sigmoid()
            ).to(x.device)

        y = self.avg_pool(x)
        y = self.fc(y)
        return x * y