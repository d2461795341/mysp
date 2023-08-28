from itertools import product
from turtle import shape
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import argparse
import clip
from collections import OrderedDict
from clip_modules.model_loader import load
from model.common import *
import numpy as np

class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.fc(x)
        return x

class MYSP(nn.Module):
    def __init__(self, config, attributes, classes, offset):
        super().__init__()
        clip_model, _ = load(config.clip_model, context_length=config.context_length)
        self.clip = clip_model
        self.config = config
        self.attributes = attributes
        self.classes = classes
        self.attr_dropout = nn.Dropout(config.attr_dropout)
        self.token_ids, self.soft_att_obj, self.ctx_vectors = self.construct_soft_prompt()
        self.freeze_soft_att_obj = self.soft_att_obj.detach().clone().cpu()
        self.freeze_ctx_vectors =  self.ctx_vectors.detach().clone().cpu()
        self.offset = offset
        self.enable_pos_emb = True
        dtype = None
        if dtype is None:
            self.dtype = torch.float16
        else:
            self.dtype = dtype
        self.text_encoder = CustomTextEncoder(self.clip, self.dtype)
        self.adapter = Adapter(768, 4)

        for name, param in self.named_parameters():
            if 'adapter' not in name:
                param.requires_grad_(False)

        self.soft_att_obj = nn.Parameter(self.soft_att_obj)
        self.soft_prompt = nn.Parameter(self.ctx_vectors).cuda()
        self.fusion = FusionTextImageBlock(config.width_img, config.width_txt, len(self.attributes), len(self.classes),
                                           config.SA_K, context_length=self.config.context_length,
                                           fusion=self.config.fusion)
        self.weight = config.res_w

    def construct_soft_prompt(self):
        token_ids = clip.tokenize("a photo of x x",
                                  context_length=self.config.context_length).cuda()

        tokenized = torch.cat(
            [
                clip.tokenize(tok, context_length=self.config.context_length)
                for tok in self.attributes + self.classes
            ]
        )
        orig_token_embedding = self.clip.token_embedding(tokenized.cuda())

        # with torch.no_grad():
        soft_att_obj = torch.zeros(
            (len(self.attributes) + len(self.classes), orig_token_embedding.size(-1)),
        )
        for idx, rep in enumerate(orig_token_embedding):
            eos_idx = tokenized[idx].argmax()
            soft_att_obj[idx, :] = torch.mean(rep[1:eos_idx, :], axis=0)

        ctx_init = "a photo of "
        n_ctx = len(ctx_init.split())
        prompt = clip.tokenize(ctx_init,
                               context_length=self.config.context_length).cuda()
        with torch.no_grad():
            embedding = self.clip.token_embedding(prompt)
        ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
        return token_ids, soft_att_obj, ctx_vectors

    def construct_token_tensors(self, pair_idx):
        attr_idx, obj_idx = pair_idx[:, 0], pair_idx[:, 1]
        class_token_ids = self.token_ids.repeat(len(pair_idx), 1)

        token_tensor = self.clip.token_embedding(
            class_token_ids.cuda()
        ).type(self.clip.dtype)

        soft_att_obj = self.attr_dropout(self.soft_att_obj)
        eos_idx = int(self.token_ids[0].argmax())

        token_tensor[:, eos_idx - 2, :] = soft_att_obj[
            attr_idx
        ].type(self.clip.dtype)
        token_tensor[:, eos_idx - 1, :] = soft_att_obj[
            obj_idx + self.offset
            ].type(self.clip.dtype)

        # adding the correct learnable context
        token_tensor[
        :, 1: len(self.soft_prompt) + 1, :
        ] = self.soft_prompt.type(self.clip.dtype)

        token_tensor011 = token_tensor.detach().clone()
        token_tensor101 = token_tensor.detach().clone()
        token_tensor110 = token_tensor.detach().clone()

        token_tensor011[
        :, 1: len(self.soft_prompt) + 1, :
        ] = self.freeze_ctx_vectors.type(self.clip.dtype)

        token_tensor101[:, eos_idx - 2, :] = self.freeze_soft_att_obj[
            attr_idx.cpu()
        ].type(self.clip.dtype)

        token_tensor110[:, eos_idx - 1, :] = self.freeze_soft_att_obj[
            obj_idx.cpu() + self.offset
            ].type(self.clip.dtype)

        return token_tensor011, token_tensor101, token_tensor110

    def visual(self, x: torch.Tensor):
        x = self.clip.visual.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.clip.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1],
                                                                                  dtype=x.dtype, device=x.device), x],
                      dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.clip.visual.positional_embedding.to(x.dtype)
        x = self.clip.visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        img_feature = self.clip.visual.transformer(x)

        x = img_feature.permute(1, 0, 2)  # LND -> NLD

        x = self.clip.visual.ln_post(x[:, 0, :])

        if self.clip.visual.proj is not None:
            x = x @ self.clip.visual.proj
        return x, img_feature

    def ft_to_logit(self, img, txt):
        img_feature = img.permute(1, 0, 2)  # LND -> NLD

        img_feature = self.clip.visual.ln_post(img_feature[:, 0, :])
        if self.clip.visual.proj is not None:
            img_feature = img_feature @ self.clip.visual.proj

        if self.config.fusion in ["BiFusion", "img2txt"]:
            txt_feature = txt.permute(0, 2, 1, 3)
            txt_feature = self.text_encoder.ln_final(txt_feature)
            txt_tf = (
                    txt_feature[
                    :, torch.arange(txt_feature.shape[1]), self.token_ids.argmax(dim=-1)
                    ]  # POS of <EOS>
                    @ self.text_encoder.text_projection
            )
        else:
            txt_feature = txt.permute(1, 0, 2)
            txt_feature = self.text_encoder.ln_final(txt_feature)
            txt_tf = (
                    txt_feature[
                        torch.arange(txt_feature.shape[0]), self.token_ids.argmax(dim=-1)
                    ]  # POS of <EOS>
                    @ self.text_encoder.text_projection
            )
        return img_feature, txt_tf

    def decompose_logits(self, logits, idx):
        att_idx, obj_idx = idx[:, 0].cpu().numpy(), idx[:, 1].cpu().numpy()
        logits_att = torch.zeros(logits.shape[0], len(self.attributes)).cuda()
        logits_obj = torch.zeros(logits.shape[0], len(self.classes)).cuda()
        for i in range(len(self.attributes)):
            logits_att[:, i] = logits[:, np.where(att_idx == i)[0]].mean(-1)
        for i in range(len(self.classes)):
            logits_obj[:, i] = logits[:, np.where(obj_idx == i)[0]].mean(-1)
        return logits_att, logits_obj

    def forward(self, batch_img, idx):
        b = batch_img.shape[0]
        l, _ = idx.shape
        batch_img, img_ft = self.visual(batch_img.type(self.clip.dtype))  ## bs * 768
        token_tensor011, token_tensor101, token_tensor110 = self.construct_token_tensors(idx)

        text_feature011, text_ft011 = self.text_encoder(
            self.token_ids,
            token_tensor011,
            enable_pos_emb=self.enable_pos_emb,
        )
        text_feature101, text_ft101 = self.text_encoder(
            self.token_ids,
            token_tensor101,
            enable_pos_emb=self.enable_pos_emb,
        )
        text_feature110, text_ft110 = self.text_encoder(
            self.token_ids,
            token_tensor110,
            enable_pos_emb=self.enable_pos_emb,
        )


        img_ft011, text_ft011 = self.fusion(img_ft.type(torch.float), text_ft011.type(torch.float), idx, b)
        img_ft011, text_ft011 = self.ft_to_logit(img_ft011.type(self.clip.dtype), text_ft011.type(self.clip.dtype))
        img_ft011 = self.adapter(img_ft011.type(torch.float)).type(self.clip.dtype)

        img_ft101, text_ft101 = self.fusion(img_ft.type(torch.float), text_ft101.type(torch.float), idx, b)
        img_ft101, text_ft101 = self.ft_to_logit(img_ft101.type(self.clip.dtype), text_ft101.type(self.clip.dtype))
        img_ft101 = self.adapter(img_ft101.type(torch.float)).type(self.clip.dtype)

        img_ft110, text_ft110 = self.fusion(img_ft.type(torch.float), text_ft110.type(torch.float), idx, b)
        img_ft110, text_ft110 = self.ft_to_logit(img_ft110.type(self.clip.dtype), text_ft110.type(self.clip.dtype))
        img_ft110 = self.adapter(img_ft110.type(torch.float)).type(self.clip.dtype)



        batch_img011 = self.weight * batch_img + (1 - self.weight) * img_ft011
        normalized_img011 = batch_img011 / batch_img011.norm(dim=-1, keepdim=True)
        text_feature011 = self.weight * text_feature011 + (1 - self.weight) * text_ft011
        idx_text_feature011 = text_feature011 / text_feature011.norm(
            dim=-1, keepdim=True
        )

        batch_img101 = self.weight * batch_img + (1 - self.weight) * img_ft101
        normalized_img101 = batch_img101 / batch_img101.norm(dim=-1, keepdim=True)
        text_feature101 = self.weight * text_feature101 + (1 - self.weight) * text_ft101
        idx_text_feature101 = text_feature101 / text_feature101.norm(
            dim=-1, keepdim=True
        )

        batch_img110 = self.weight * batch_img + (1 - self.weight) * img_ft110
        normalized_img110 = batch_img110 / batch_img110.norm(dim=-1, keepdim=True)
        text_feature110 = self.weight * text_feature110 + (1 - self.weight) * text_ft110
        idx_text_feature110 = text_feature110 / text_feature110.norm(
            dim=-1, keepdim=True
        )

        logits011 = (
                    self.clip.logit_scale.exp()
                    * normalized_img011
                    @ idx_text_feature011.t()
            )

        logits101 = (
                self.clip.logit_scale.exp()
                * normalized_img101
                @ idx_text_feature101.t()
        )

        logits110 = (
                self.clip.logit_scale.exp()
                * normalized_img110
                @ idx_text_feature110.t()
        )

        logits = (logits011+logits101+logits110)/3

        logits_att, logits_obj = self.decompose_logits(logits, idx)

        return (logits, logits_att, logits_obj)

