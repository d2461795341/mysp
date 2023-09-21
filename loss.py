from torch.nn.modules.loss import CrossEntropyLoss
import torch.nn as nn
import torch

def loss_calu(predict, target, config):
    loss_fn = CrossEntropyLoss()
    batch_img, batch_attr, batch_obj, batch_target = target
    batch_attr = batch_attr.cuda()
    batch_obj = batch_obj.cuda()
    batch_target = batch_target.cuda()
    logits_c, logits_c2s, logits_c2o, logits_soft_prompt = predict
    loss_logit_df = loss_fn(logits_c, batch_target)

    loss_logit_sp = loss_fn(logits_soft_prompt, batch_target)

    loss_c2s = loss_fn(logits_c2s, batch_attr)
    loss_c2o = loss_fn(logits_c2o, batch_obj)

    loss = loss_logit_df + config.com2_att_obj_w * (loss_c2s + loss_c2o) + config.sp_w * loss_logit_sp

    return loss