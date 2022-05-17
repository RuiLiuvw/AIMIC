from typing import Any

from torch import nn, Tensor


class BaseConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        BNorm: bool = False,
        ActType: str = '',
        **kwargs: Any
    ) -> None:
        super(BaseConv2d, self).__init__()
        self.Conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.Bn = nn.BatchNorm2d(out_channels, eps=0.001) if BNorm else nn.Identity()
        if ActType == '':
            self.Act = nn.ReLU(inplace=True)  
        elif ActType == 'swish':
            self.Act = nn.SiLU(inplace=True)
        else:
            self.Act = nn.Identity()
        
        
    def forward(self, x: Tensor) -> Tensor:
        x = self.Conv(x)
        x = self.Bn(x)
        return self.Act(x)

    
def initWeight(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, 0.01)
        nn.init.constant_(m.bias, 0)
    