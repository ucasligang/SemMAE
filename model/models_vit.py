import torch
import torch.nn as nn
from functools import partial
import model.vision_transformer

class VisionTransformer(model.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def forward_features(self, x):
        x = x.squeeze()
        B,patches_num = x.shape[0], x.shape[1]
        sorted_index = x[:, :, 14:15].type(torch.int64)
        # posintional_embed = x[:, :, 15:783]
        patches = x[:, :, 15:].type(torch.cuda.FloatTensor)  # [64, 196, 207]
        img = self.unpatchify(patches)  # [64, 3, 112, 112]
        x = self.patch_embed(img)
        pos_embeddings = self.pos_embed[:, 1:, :].squeeze()  # [784, 768]
        sorted_index = sorted_index.reshape(-1)  # [64*196]
        pos_embeddings = torch.index_select(pos_embeddings, dim=0,
                                            index=sorted_index)  # [sorted_index.type(torch.long)]
        pos_embeddings = pos_embeddings.reshape(B, patches_num, -1)  # [64, 196, 768]
        # add pos embed w/o cls token
        # x = x + self.pos_embed[:, 1:, :]
        x = x + pos_embeddings  # [64, 196, 768]
        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome

def vit_base_patch8(**kwargs):
    model = VisionTransformer(
        img_size=112, patch_size=8, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
