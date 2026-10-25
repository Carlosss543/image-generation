import torch
import torch.nn as nn
import math


class TimestepEmbedder(nn.Module):

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()

        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t*1000.0, self.frequency_embedding_size) # with flow matching t is in [0, 1] so we scale it to [0, 1000] for the embedding
        t_emb = self.mlp(t_freq)
        return t_emb


class TransformerBlock(nn.Module):

    def __init__(self, embed_dim, num_heads, dropout):
        super().__init__()

        self.ln1 = nn.LayerNorm(embed_dim, elementwise_affine=False)
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(embed_dim, elementwise_affine=False)
        self.ffwd = nn.Sequential(
            nn.Linear(embed_dim, 4*embed_dim),
            nn.GELU(),
            nn.Linear(4*embed_dim, embed_dim),
            nn.Dropout(dropout)
        )
        # scale and shift parameter for ln1 (2*embed_dim)
        # scale and shift parameter for ln2 (2*embed_dim)
        # scale for output of attention prior to residual connection (embed_dim)
        # scale for output of mlp prior to residual connection (embed_dim)
        # total : (6, embed_dim)
        self.adaLN_zero = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), # I added this extra linear but it is not present in the DiT repo
            nn.SiLU(),
            nn.Linear(embed_dim, 6*embed_dim)
        )
        # initialized at 0 to have the identity function at initialization (all residual connections at 0)
        nn.init.constant_(self.adaLN_zero[2].weight, 0)
        nn.init.constant_(self.adaLN_zero[2].bias, 0)

    def forward(self, x, condition):
        scale_shift_params = self.adaLN_zero(condition).unsqueeze(1).chunk(6, dim=-1) # returns a tuple of the 6 parameters
        pre_attn_scale, pre_attn_shift, post_attn_scale, pre_mlp_scale, pre_mlp_shift, post_mlp_scale = scale_shift_params
    
        x1 = self.ln1(x) * (1 + pre_attn_scale) + pre_attn_shift
        x1, _ = self.mha(x1, x1, x1, need_weights=False)
        x = x + x1 * post_attn_scale

        x2 = self.ln2(x) * (1 + pre_mlp_scale) + pre_mlp_shift
        x2 = self.ffwd(x2)
        x = x + x2 * post_mlp_scale
        
        return x


class DiT(nn.Module):

    def __init__(self, img_size, img_channels, num_classes, patch_size, embed_dim, num_blocks, num_heads, dropout):
        super().__init__()

        self.patch_size = patch_size
        self.img_channels = img_channels

        self.patchify = nn.Conv2d(img_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embedding = nn.Parameter(torch.empty(1, (img_size//patch_size)**2, embed_dim).normal_(std=0.02))

        self.t_embedding = TimestepEmbedder(embed_dim)
        if num_classes is not None:
            self.label_embedding_table = nn.Embedding(num_classes, embed_dim)

        self.transformer_blocks = nn.ModuleList([TransformerBlock(embed_dim, num_heads, dropout) for _ in range(num_blocks)])

        self.final_ln = nn.LayerNorm(embed_dim, elementwise_affine=False)
        self.unpatchify = nn.Linear(embed_dim, patch_size**2 * img_channels)

        # scale and shift for the final layernorm
        self.adaLN_zero_unpatchify = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, 2*embed_dim)
        )
        nn.init.constant_(self.adaLN_zero_unpatchify[2].weight, 0)
        nn.init.constant_(self.adaLN_zero_unpatchify[2].bias, 0)

    def forward(self, imgs, t, labels=None):
        token_embed = self.patchify(imgs).flatten(2).transpose(1, 2)
        x = token_embed + self.pos_embedding # (B, num_patches, embed_dim)

        embed = self.t_embedding(t)
        if labels is not None:
            embed += self.label_embedding_table(labels)

        for block in self.transformer_blocks:
            x = block(x, embed)

        pre_ln_scale, pre_ln_shift = self.adaLN_zero_unpatchify(embed).unsqueeze(1).chunk(2, dim=-1)
        x = self.final_ln(x) * (1 + pre_ln_scale) + pre_ln_shift

        x = self.unpatchify(x) # (B, num_patches, patch_size**2 * C)
        x = x.view(-1, imgs.shape[2]//self.patch_size, imgs.shape[3]//self.patch_size, self.img_channels, self.patch_size, self.patch_size) # (B, H//patch_size, W//patch_size, C, patch_size, patch_size)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous() # (B, C, H//patch_size, patch_size, W//patch_size, patch_size)
        x = x.view(-1, self.img_channels, imgs.shape[2], imgs.shape[3]) # (B, C, H, W)

        return x


def dit(img_size, img_channels, num_classes):
    return DiT(
        img_size=img_size,
        img_channels=img_channels,
        num_classes=num_classes,
        patch_size=2,
        num_blocks=12,
        num_heads=12,
        embed_dim=12*64,
        dropout = 0.0
    )
