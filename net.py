import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer import SwinTransformer
from get_args import get_args
from something import SpaceAttention, ChannelAttentionLayer

def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb

def nonlinearity(x):
    # swish
    return x * torch.sigmoid(x)

def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(
            x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x

class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        self.temb_proj = torch.nn.Linear(temb_channels,
                                         out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h

class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)  # b,hw,c
        k = k.reshape(b, c, h * w)  # b,c,hw
        w_ = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = torch.bmm(v, w_)
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x + h_

class feature_map(nn.Module):
    def __init__(self, in_channel, ch):
        super().__init__()

        self.feature_conv1 = torch.nn.Conv2d(in_channel,
                                             ch,
                                             kernel_size=3,
                                             stride=2,
                                             padding=1)
        self.feature_conv2 = torch.nn.Conv2d(ch,
                                             ch * 2,
                                             kernel_size=3,
                                             stride=2,
                                             padding=1)
        self.feature_conv3 = torch.nn.Conv2d(ch * 2,
                                             ch * 4,
                                             kernel_size=3,
                                             stride=2,
                                             padding=1)
        self.feature_conv4 = torch.nn.Conv2d(ch * 4,
                                             ch * 8,
                                             kernel_size=3,
                                             stride=2,
                                             padding=1)
        self.leaky_relu = nn.LeakyReLU(0.2)


    def forward(self, map):
        fs = []
        fout1 = self.leaky_relu(self.feature_conv1(map))
        fs.append(fout1)
        fout2 = self.leaky_relu(self.feature_conv2(fout1))
        fs.append(fout2)
        fout3 = self.leaky_relu(self.feature_conv3(fout2))
        fs.append(fout3)
        fout4 = self.leaky_relu(self.feature_conv4(fout3))
        fs.append(fout4)

        return fs


class AttnChannel(nn.Module):
    def __init__(self, n_feat, bias=False, groups=1):
        super(AttnChannel, self).__init__()

        act = nn.LeakyReLU(0.2)

        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=1, padding=1, bias=bias, groups=groups),
            act,
            nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=1, padding=1, bias=bias, groups=groups)
        )

        self.act = act

        self.gcnet = nn.Sequential(
                ChannelAttentionLayer(n_feat, reduction=4, feature=None),
                SpaceAttention()
            )

    def forward(self, x):
        res = self.body(x)
        res = self.act(self.gcnet(res))
        res += x
        return res


class time_embed(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.ch = ch
        self.temb_ch = self.ch * 4
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            torch.nn.Linear(self.ch,
                            self.temb_ch),
            torch.nn.Linear(self.temb_ch,
                            self.temb_ch),
        ])

    def forward(self, t):
        # timestep embedding
        temb = get_timestep_embedding(t, self.ch)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)

        return temb


class DiffModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.ch = args.ch
        self.in_channel = args.in_ch
        self.feature_in_channel = args.fea_in_ch
        self.out_ch = args.out_ch
        self.dropout = args.dropout
        self.attn_resolutions = args.attn
        self.resamp_with_conv = args.resamp_with_conv
        self.temb_ch = self.ch * 4
        self.num_res_blocks = args.num_res_blocks

        curr_res = args.image_size
        ch_mult = tuple(args.ch_mult)
        in_ch_mult = (1,) + ch_mult
        self.num_resolutions = len(ch_mult)


        self.time_embed = time_embed(self.ch)

        # original
        self.feature_map = feature_map(self.feature_in_channel,
                                       self.ch)

        self.transformer = SwinTransformer(img_size=224, patch_size=1, in_chans=3, num_classes=1, drop_path_rate=0.2)

        # IN and out
        self.in_out = nn.ModuleList()
        self.in_out.conv_in = torch.nn.Conv2d(self.in_channel, self.ch, kernel_size=3, stride=2, padding=1)

        self.channelattn = nn.ModuleList()
        self.channelattn.up = nn.ModuleList([AttnChannel(self.ch * ch_mult[i]) for i in range(self.num_resolutions)])
        self.channelattn.down = nn.ModuleList([AttnChannel(2 * self.ch * ch_mult[j]) for j in range(self.num_resolutions)])

        # down sample module
        self.down = nn.ModuleList()
        block_in = None
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = self.ch * in_ch_mult[i_level]
            block_out = self.ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=self.dropout))
                block_in = block_out
                if curr_res in self.attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = DownsampleBlock(block_in, self.resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=self.dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=self.dropout)

        # upsampling

        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = self.ch * ch_mult[i_level]
            skip_in = self.ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                if i_block == self.num_res_blocks:
                    skip_in = self.ch * in_ch_mult[i_level]
                block.append(ResnetBlock(in_channels=block_in + skip_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=self.dropout))
                block_in = block_out
                if curr_res in self.attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = UpsampleBlock(block_in, self.resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        #  in and OUT
        self.in_out.norm_out = Normalize(block_in)
        self.in_out.conv_out = torch.nn.ConvTranspose2d(block_in, self.out_ch, kernel_size=4, stride=2, padding=1)



    def forward(self, low, xt, illu, reflect, t):
        x = torch.cat([xt, low], dim=1)

        temb = self.time_embed(t)
        fs = self.feature_map(illu)

        ts = self.transformer(reflect)


        # downsampling
        hs = [self.in_out.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                fea_down = fs[i_level] * ts[i_level] + hs[-1]
                fea_down_attn = self.channelattn.up[i_level](fea_down)
                hs.append(self.down[i_level].downsample(fea_down_attn))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)


        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](
                    torch.cat([h, hs.pop()], dim=1), temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                fea_up = fs[i_level] * ts[i_level] + h
                fea_up_attn = self.channelattn.up[i_level](fea_up)
                h = self.up[i_level].upsample(fea_up_attn)

        h = self.in_out.norm_out(h)
        h = nonlinearity(h)
        h = self.in_out.conv_out(h)

        return h


