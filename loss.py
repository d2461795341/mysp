from torch.nn.modules.loss import CrossEntropyLoss
import torch.nn as nn
import torch

def loss_calu(predict, target, config):
    #loss_fn = CrossEntropyLoss()
    loss_fn = nn.NLLLoss()
    batch_img, batch_attr, batch_obj, batch_target = target
    batch_attr = batch_attr.cuda()
    batch_obj = batch_obj.cuda()
    batch_target = batch_target.cuda()

    #logits, logits_att, logits_obj, logits_soft_prompt = predict
    #loss_logit_df = loss_fn(logits, batch_target)
    #loss_logit_sp = loss_fn(logits_soft_prompt, batch_target)
    #loss_att = loss_fn(logits_att, batch_attr)
    #loss_obj = loss_fn(logits_obj, batch_obj)
    #loss = loss_logit_df + config.att_obj_w * (loss_att + loss_obj) + config.sp_w * loss_logit_sp

    prob, prob_att, prob_obj = predict
    loss_logit_df = loss_fn(torch.log(prob+1e-7), batch_target)
    loss_att = loss_fn(torch.log(prob_att+1e-7), batch_attr)
    loss_obj = loss_fn(torch.log(prob_obj+1e-7), batch_obj)
    loss = loss_logit_df + config.att_obj_w * (loss_att + loss_obj)

    return loss