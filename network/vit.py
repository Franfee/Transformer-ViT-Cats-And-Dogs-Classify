import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),  # 进入前LN正则化,接着attention层
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))  # 进入前LN正则化,接着FFN层
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x  # 残差连接Add
            x = ff(x) + x  # 残差连接Add
        return x


class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool='cls', channels=3,
                 dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, \
            'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width  # token纬度,论文page3
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(  # 论文中page3蓝色字: X*E=[196x768]*[768x768]=[196x768]
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim, dim),  # 架构图中 Linear Projection of Flattened Patches
        )
        # 与transformer论文不同的点,可学习的位置编码
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))  # 论文中[1 x (196+1) x 768]
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))  # 论文中[1x1x768]
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)  # transformer论文的编码器

        self.pool = pool  # cls汇聚方式:             数据使用cls[batch x数据第0号token]
        self.to_latent = nn.Identity()  # 不区分参数的占位符标识运算符,可以放到残差网络里就是在跳过连接的地方用这个层,此处仅占位

        self.mlp_head = nn.Sequential(  # 解码器,由于任务简单,使用mlp,数据使用cls[batch x数据第0号token]
            nn.LayerNorm(dim),  # 论文架构图 橙色MLP Head
            nn.Linear(dim, num_classes)  # 论文架构图 橙色Class
        )

    def forward(self, img):
        """

        :param img: [B C H W]
        :return:
        """
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape  # 论文中x=[Bx196x768]

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)  # 论文中cls=[Bx1x768]
        x = torch.cat((cls_tokens, x), dim=1)  # 论文中XE=[(196+1)x768],x=[Bx(196+1)x768]
        x += self.pos_embedding[:, :(n + 1)]  # 直接加
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)


if __name__ == '__main__':
    DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    model = ViT(image_size=224, patch_size=32, num_classes=10, dim=128, depth=12, heads=12, mlp_dim=128).to(DEVICE)
    x = torch.rand(1, 3, 224, 224).to(DEVICE)
    out = model(x)
    print(x.shape, out.shape)
