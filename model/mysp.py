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
from model.mycommon import *
import numpy as np
from utils import stable_softmax

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
        orig_type = x.dtype
        x = self.fc(x.type(torch.float32))
        return x.type(orig_type)

class MYSP(nn.Module):
    def __init__(self, config, attributes, classes, offset):
        super().__init__()
        clip_model, _ = load(config.clip_model, context_length=config.context_length)
        self.clip = clip_model
        self.config = config
        self.attributes = attributes
        self.classes = classes
        self.attr_dropout = nn.Dropout(config.attr_dropout)
        self.token_ids_s, self.token_ids_o, self.token_ids_c, self.soft_att_obj, self.ctx_vectors_s, self.ctx_vectors_o, self.ctx_vectors_c = self.construct_soft_prompt()

        self.offset = offset
        self.enable_pos_emb = True
        dtype = None
        if dtype is None:
            self.dtype = torch.float16
        else:
            self.dtype = dtype
        self.text_encoder = CustomTextEncoder(self.clip, self.dtype)

        #adapter部分还可以继续优化
        if(self.config.adapter_place!="none"):
            self.additional_visual_params = nn.ModuleList([Adapter(1024,4) for i in range(2*self.clip.visual.transformer.layers)])

        for name, param in self.named_parameters():
            if 'adapter' not in name:
                param.requires_grad_(False)

        self.soft_att_obj = nn.Parameter(self.soft_att_obj)
        self.soft_prompt_s = nn.Parameter(self.ctx_vectors_s).cuda()
        self.soft_prompt_o = nn.Parameter(self.ctx_vectors_o).cuda()
        self.soft_prompt_c = nn.Parameter(self.ctx_vectors_c).cuda()
        #经过projection后width_img变为width_txt
        self.fusion = FusionTextImageBlock(config.width_img, config.width_txt, len(self.attributes), len(self.classes),
                                           config.SA_K, context_length=self.config.context_length,
                                           fusion=self.config.fusion)
        self.weight = config.res_w

    def construct_soft_prompt(self):
        token_ids_s = clip.tokenize("a photo of x",
                                  context_length=self.config.context_length).cuda()
        token_ids_o = clip.tokenize("a photo of x",
                                  context_length=self.config.context_length).cuda()
        token_ids_c = clip.tokenize("a photo of x x",
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
        ctx_vectors_s = ctx_vectors.detach().clone()
        ctx_vectors_o = ctx_vectors.detach().clone()
        ctx_vectors_c = ctx_vectors.detach().clone()
        return token_ids_s, token_ids_o, token_ids_c, soft_att_obj, ctx_vectors_s,  ctx_vectors_o, ctx_vectors_c

    def construct_token_tensors(self, pair_idx):
        attr_idx, obj_idx = pair_idx[:, 0], pair_idx[:, 1]

        state_idx, object_idx = torch.tensor([i for i in range(len(self.attributes))]).cuda(), torch.tensor([i for i in range(len(self.classes))]).cuda()

        state_token_ids = self.token_ids_s.repeat(len(self.attributes), 1)
        object_token_ids = self.token_ids_o.repeat(len(self.classes), 1)
        class_token_ids = self.token_ids_c.repeat(len(pair_idx), 1)

        token_tensor_s = self.clip.token_embedding(
            state_token_ids.cuda()
        ).type(self.clip.dtype)

        token_tensor_o = self.clip.token_embedding(
            object_token_ids.cuda()
        ).type(self.clip.dtype)

        token_tensor_c = self.clip.token_embedding(
            class_token_ids.cuda()
        ).type(self.clip.dtype)


        soft_att_obj = self.attr_dropout(self.soft_att_obj)

        eos_idx_s = int(self.token_ids_s[0].argmax())
        eos_idx_o = int(self.token_ids_o[0].argmax())
        eos_idx_c = int(self.token_ids_c[0].argmax())

        token_tensor_s[:, eos_idx_s - 1, :] = soft_att_obj[
           state_idx
        ].type(self.clip.dtype)
        token_tensor_s[
        :, 1: len(self.soft_prompt_s) + 1, :
        ] = self.soft_prompt_s.type(self.clip.dtype)


        token_tensor_o[:, eos_idx_o - 1, :] = soft_att_obj[
            object_idx + self.offset
        ].type(self.clip.dtype)
        token_tensor_o[
        :, 1: len(self.soft_prompt_o) + 1, :
        ] = self.soft_prompt_o.type(self.clip.dtype)


        token_tensor_c[:, eos_idx_c - 2, :] = soft_att_obj[
            attr_idx
        ].type(self.clip.dtype)
        token_tensor_c[:, eos_idx_c - 1, :] = soft_att_obj[
            obj_idx + self.offset
            ].type(self.clip.dtype)
        token_tensor_c[
        :, 1: len(self.soft_prompt_c) + 1, :
        ] = self.soft_prompt_c.type(self.clip.dtype)

        return token_tensor_s, token_tensor_o, token_tensor_c

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

        #DFSP version
        
        img_feature = self.clip.visual.transformer(x)
        x = img_feature.permute(1, 0, 2)  # LND -> NLD

        x = self.clip.visual.ln_post(x[:, 0, :])

        if self.clip.visual.proj is not None:
            x = x @ self.clip.visual.proj
        return x, img_feature
        
        #Troika version
        '''
        x = self.clip.visual.transformer(x)
        img_feature = x.permute(1, 0, 2)  # LND -> NLD

        img_feature = self.clip.visual.ln_post(img_feature)
        if self.clip.visual.proj is not None:
            img_feature = img_feature @ self.clip.visual.proj
        return img_feature[:, 0, :], img_feature
        '''

    def encode_image_with_adapter(self, x: torch.Tensor):
        # self.clip is the CLIP model
        x = self.clip.visual.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.clip.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.clip.visual.positional_embedding.to(x.dtype)
        x = self.clip.visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        
        # img_feature = self.clip.visual.transformer(x)
        for i_block in range(self.clip.visual.transformer.layers):
            # add Adapter to self.attention
            if self.config.adapter_place in ['attn', 'all'] and self.config.adapter_option == 'parallel':
                # self.additional_visual_params is nn.ModuleList that contains Adapters for each layer
                adapt_x = self.additional_visual_params[i_block](x)
            residual = x
            x = self.clip.visual.transformer.resblocks[i_block].attention(
                self.clip.visual.transformer.resblocks[i_block].ln_1(x)
            )
            if self.config.adapter_place in ['attn', 'all']:
                if self.config.adapter_option == 'parallel':
                    x = x + adapt_x
                elif self.config.adapter_option == 'sequential':
                    x = self.additional_visual_params[i_block](x)
                else:
                    raise NotImplementedError('{0} is not an implemented adapter option!'.format(self.config.adapter_option))
            x = x + residual

            # add Adapter to feed-forward
            i_adapter = i_block
            if self.config.adapter_place == 'all':
                i_adapter = i_adapter + self.clip.visual.transformer.layers
            if self.config.adapter_place in ['ffn', 'all'] and self.config.adapter_option == 'parallel':
                adapt_x = self.additional_visual_params[i_adapter](x)
            residual = x
            x = self.clip.visual.transformer.resblocks[i_block].mlp(
                self.clip.visual.transformer.resblocks[i_block].ln_2(x)
            )
            if self.config.adapter_place in ['ffn', 'all']:
                if self.config.adapter_option == 'parallel':
                    x = x + adapt_x
                elif self.config.adapter_option == 'sequential':
                    x = self.additional_visual_params[i_adapter](x)
                else:
                    raise NotImplementedError('{0} is not an implemented adapter option!'.format(self.config.adapter_option))
            x = x + residual

        img_feature = x.permute(1, 0, 2)  # LND -> NLD

        img_feature = self.clip.visual.ln_post(img_feature)
        if self.clip.visual.proj is not None:
            img_feature = img_feature @ self.clip.visual.proj
        return img_feature[:, 0, :], img_feature

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
                        torch.arange(txt_feature.shape[0]), self.token_ids_c.argmax(dim=-1)
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
        if(self.config.adapter_place=="none"):
            batch_img, img_ft = self.visual(batch_img.type(self.clip.dtype))  ## bs * 768
        else:
            batch_img, img_ft = self.encode_image_with_adapter(batch_img.type(self.clip.dtype))  ## bs * 768
        
        token_tensor_s, token_tensor_o, token_tensor_c = self.construct_token_tensors(idx)

        text_feature_s, text_ft_s = self.text_encoder(
            self.token_ids_s,
            token_tensor_s,
            enable_pos_emb=self.enable_pos_emb,
        )
        text_feature_o, text_ft_o = self.text_encoder(
            self.token_ids_o,
            token_tensor_o,
            enable_pos_emb=self.enable_pos_emb,
        )
        text_feature_c, text_ft_c = self.text_encoder(
            self.token_ids_c,
            token_tensor_c,
            enable_pos_emb=self.enable_pos_emb,
        )


        img_ft, text_ft_c = self.fusion(img_ft.type(torch.float), text_ft_s.type(torch.float), text_ft_o.type(torch.float), text_ft_c.type(torch.float), idx, b)
        

        img_ft, text_ft_c = self.ft_to_logit(img_ft.type(self.clip.dtype), text_ft_c.type(self.clip.dtype))


        batch_img_soft_prompt = batch_img / batch_img.norm(dim=-1, keepdim=True)
        text_features_soft_prompt = text_feature_c / text_feature_c.norm(dim=-1, keepdim=True)


        batch_img = self.weight * batch_img + (1 - self.weight) * img_ft
        normalized_img = batch_img / batch_img.norm(dim=-1, keepdim=True)
        text_feature_c = self.weight * text_feature_c + (1 - self.weight) * text_ft_c
        idx_text_feature_c = text_feature_c / text_feature_c.norm(dim=-1, keepdim=True)


        logits_c = (
                    self.clip.logit_scale.exp()
                    * normalized_img
                    @ idx_text_feature_c.t()
            )
        
        
        logits_soft_prompt = (
            self.clip.logit_scale.exp()
            * batch_img_soft_prompt
            @ text_features_soft_prompt.t()
        )     


        logits_c2s, logits_c2o = self.decompose_logits(logits_c, idx)




        return (logits_c, logits_c2s, logits_c2o, logits_soft_prompt)

