import torch 
import torch as th 
import torch.nn as nn
import torch.nn.functional as tnf 
import torchaudio.transforms as tt 

class DWConv(nn.Module):
    '''DepthWise convolution
    '''
    def __init__(self,
                 type: str,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: tuple,
                 dilation: tuple=(1,1),
                 padding: int=0,
                ) -> None:
        super().__init__()
        
        conv = {
            "one": nn.Conv1d,
            "two": nn.Conv2d
        }

        self.depthwise = conv[type](in_channels, in_channels, kernel_size, padding=padding, dilation=dilation, groups=in_channels)
        self.pointwise = conv[type](in_channels, out_channels, kernel_size=1, padding=padding, dilation=dilation)
    
    def forward(self, x: th.Tensor) -> th.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x 


class SubSpectralNorm(nn.Module):
    '''SubSpectralNorm(SSN) to replace BN
    '''
    def __init__(self,
        num_features: int,
        num_subspecs: int=2,
        eps: float=1e-5,
        affine: bool=True
    ):
        super(SubSpectralNorm, self).__init__()
        self.eps = eps
        self.subpecs = num_subspecs
        self.gamma = nn.Parameter(torch.ones(1,num_features * num_subspecs,1,1),
            requires_grad=affine)
        self.beta = nn.Parameter(torch.zeros(1,num_features * num_subspecs,1,1),
            requires_grad=affine)
    
    def forward(self, x: th.Tensor) -> th.Tensor:
        batch_size, num_channels, height, width = x.size()
        # x = x.view(batch_size, num_channels*self.subpecs, height//self.subpecs, width)
        
        x = x.view(batch_size, num_channels*self.subpecs, height//self.subpecs, -1)

        x_mean = x.mean([0, 2, 3]).view(1, num_channels * self.subpecs, 1, 1)
        x_var = x.var([0, 2, 3]).view(1, num_channels * self.subpecs, 1, 1)

        x = (x - x_mean) / (x_var + self.eps).sqrt() * self.gamma + self.beta

        return x.view(batch_size, num_channels, height, width)



class NormalBlock(nn.Module):
    def __init__(self,
                 feat_dim: int,
                 dilation: tuple=(1,1),
                ) -> None:
        super().__init__()

        self.f2 = nn.Sequential(
                    DWConv("two", feat_dim, feat_dim, (3,1)), # frequency-depthwise convolution
                    # SubSpectralNorm(feat_dim, 4)                    
                    nn.BatchNorm2d(feat_dim)
        )

        self.f1 = nn.Sequential(
                    DWConv("one", feat_dim, feat_dim, 3, dilation), # temporal-depthwise convolution
                    nn.BatchNorm1d(feat_dim),
                    nn.SiLU(),
                    nn.Conv1d(feat_dim, feat_dim, 1),
                    nn.Dropout(0.1)
        )

        self.aux = nn.Conv2d(feat_dim, feat_dim, (1,1))

        self.act = nn.ReLU()

    def forward(self, x: th.Tensor) -> th.Tensor:
        '''
        x: [batch, channel, frequency, time]
        return: [batch, channel, frequency, time]
        '''
        B, C, F, T = x.shape        
        ori = x 
        x = self.f2(x)
        res = self.aux(x)

        # [batch, channel, frequency, time] -> [batch, channel, 1, time]
        # -> [batch, channel, time]
        x = tnf.avg_pool2d(x, (x.shape[2],1))
        x = x.squeeze(2)
        x = self.f1(x)

        # [batch, channel, time] -> [batch, channel, 1, time]
        # -> [batch, channel, frequency, time]
        x = x.unsqueeze(2)
        x = x.repeat(1, 1, F, 1)

        if x.shape[2] != ori.shape[2]:
            x = tnf.pad(x, [0, 0, 0, abs(x.shape[2]-ori.shape[2])])
        if x.shape[-1] != ori.shape[-1]:
            x = tnf.pad(x, [0, abs(x.shape[-1]-ori.shape[-1]), 0, 0])        
        if res.shape[2] != ori.shape[2]:
            res = tnf.pad(res, [0, 0, 0, abs(res.shape[2]-ori.shape[2])])

        x = x + ori + res
        x = self.act(x)
        return x 


class TransitionBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: tuple=(1,1),
                 dilation: tuple=(1,1),
                 padding: tuple=(0,0)                
                ) -> None:
        super().__init__()

        self.f2 = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, (1,1), stride, padding),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                    DWConv("two", out_channels, out_channels, (3,1)), # frequency-depthwise convolution
                    # SubSpectralNorm(out_channels, 4)
                    nn.BatchNorm2d(out_channels)
        )

        self.f1 = nn.Sequential(
                    DWConv("one", out_channels, out_channels, 3, dilation[1]), # temporal-depthwise convolution
                    nn.BatchNorm1d(out_channels),
                    nn.SiLU(),
                    nn.Conv1d(out_channels, out_channels, 1),
                    nn.Dropout(0.1)
        )

        self.act = nn.ReLU()
    
    def forward(self, x:th.Tensor) -> th.Tensor:
        x = self.f2(x)
        res = x 

        B, C, F, T = x.shape
        x = tnf.avg_pool2d(x, (x.shape[2],1))
        x = x.squeeze(2)
        x = self.f1(x)
        x = x.unsqueeze(2)
        x = x.repeat(1, 1, F, 1)

        if x.shape[-1] != T:
            x = tnf.pad(x, [0, abs(T-x.shape[-1])])

        x = x + res 
        x = self.act(x)
        return x 


class FeatureExactor(nn.Module):
    def __init__(self,
                input_freq: int=16000,
                n_fft: int=480,
                win_length: int=480, 
                hop_length: int=160,
                n_mels: int=40,
                stretch_factor: float=0.8,
                aug: bool=True
                ) -> None:
        super().__init__()

        self.spec = tt.Spectrogram(n_fft=n_fft, win_length=win_length, hop_length=hop_length, window_fn=torch.hamming_window, power=2)

        self.spec_aug = nn.Sequential(
                        tt.TimeStretch(stretch_factor, fixed_rate=True),
                        tt.FrequencyMasking(freq_mask_param=80),
                        tt.TimeMasking(time_mask_param=80),
        )

        self.mel_scale = tt.MelScale(n_mels=n_mels, sample_rate=input_freq, n_stft=n_fft //2 + 1)

        self.aug = aug 

    def forward(self, x: th.Tensor) -> th.Tensor:
        spec = self.spec(x)
        if self.aug:
            spec = self.spec_aug(spec)
        mel = self.mel_scale(spec)
        return mel 


class BCResNet(nn.Module):
    def __init__(self,
                num_classes: int,
                delta: int=1,
                aug: bool=True
                ) -> None:
        super().__init__()
        self.features = FeatureExactor(aug=aug)

        self.conv1 = nn.Sequential(
                    nn.Conv2d(1, 16*delta, (5,5), (2,1), (2,2)),
                    nn.BatchNorm2d(16*delta),
                    nn.ReLU()
        )

        self.bc1 = nn.Sequential(
                    TransitionBlock(16*delta, 8*delta, (1,1), (1,1), (1,0)),
                    NormalBlock(8*delta, 1)
        )

        self.bc2 = nn.Sequential(
                    TransitionBlock(8*delta, 12*delta, (2,1), (1,2), (2,0)),
                    NormalBlock(12*delta, 1)
        )

        self.bc3 = nn.Sequential(
                    TransitionBlock(12*delta, 16*delta, (2,1), (1,4), (2,0)),
                    NormalBlock(16*delta, 4),
                    NormalBlock(16*delta, 4),
                    NormalBlock(16*delta, 4),
        )

        self.bc4 = nn.Sequential(
                    TransitionBlock(16*delta, 20*delta, (1,1), (1,8), (1,0)),
                    NormalBlock(20*delta, 8),
                    NormalBlock(20*delta, 8),
                    NormalBlock(20*delta, 8),
        )

        self.conv2 = DWConv("two", 20*delta, 20*delta, (5,5), (1,1))
        
        self.conv3 = nn.Sequential(
                        nn.Conv2d(20*delta, 32*delta, (1,1), (1,1), dilation=(1,1)),
                        nn.BatchNorm2d(32*delta),
                        nn.ReLU()
        )

        self.conv4 = nn.Conv2d(32*delta, num_classes, (1,1))

    def forward(self, x:th.Tensor) -> th.Tensor:
        '''
        x: [batch, length] -> [batch, 1, frequency, time]
        x: [batch, num_classes]
        '''
        x = self.features(x).unsqueeze(1)
        x = self.conv1(x)
        x = self.bc1(x)
        x = self.bc2(x)
        x = self.bc3(x)
        x = self.bc4(x)
        x = self.conv2(x)
        x = self.conv3(x)

        B, C, F, T = x.shape
        x = tnf.avg_pool2d(x, (1,T))
        x = self.conv4(x)

        x = x.view(B,-1)
        return x 


if __name__  == "__main__":
    import torchaudio
    wav = th.rand([4, 16000*4])
    net = BCResNet(num_classes=6, delta=8)
    print('Number of model parameters: {:.2f} M'.format(
        sum([p.data.nelement() for p in net.parameters()]) / 1e6))    
    print(net(wav).shape)