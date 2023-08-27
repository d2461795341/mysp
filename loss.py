from torch.nn.modules.loss import CrossEntropyLoss
import torch

def loss_calu(predict, target, config):
    loss_fn = CrossEntropyLoss()
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

    logits, logits_att, logits_obj = predict
    loss_logit_df = loss_fn(logits, batch_target)
    loss_att = loss_fn(logits_att, batch_attr)
    loss_obj = loss_fn(logits_obj, batch_obj)
    loss = loss_logit_df + config.att_obj_w * (loss_att + loss_obj)

    return loss


t1 = torch.ones(3, 3)
t2 = t1/3
t1[:, 2] = 2

print(f"id(t1) = {id(t1)}")
print(f"id(t2) = {id(t2)}")
print(f"t1 = {t1}")
print(f"t2 = {t2}")