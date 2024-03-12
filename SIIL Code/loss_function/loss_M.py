import torch
import torch.nn.functional as F

def contrast_loss(embedding, target, present_cls, temperature):
    cls = embedding.shape[0]
    full_cls, queue, dim = target.shape
    idx = torch.where(present_cls == 1)[0]
    embedding = F.normalize(embedding.detach(), dim=1)
    target = F.normalize(target.detach(), dim=2)
    contrastive_loss = torch.zeros(1, dtype=torch.float32).cuda()
    for j in range(cls):
        # --------------negative sample-------------------
        similarity_neg = torch.einsum('c,nc->n', [embedding[j], target.view(-1, dim).detach()])
        logit_neg = torch.div(similarity_neg, temperature)
        max_log = torch.max(logit_neg)
        exp_logit_neg = torch.exp(logit_neg - max_log.detach())
        dist = torch.abs(torch.arange(0, 26).cuda() - idx[j]).expand(32, 26).permute(1, 0).reshape(32 * 26)
        dist = torch.exp(1 / dist)

        # --------------positive sample-------------------
        feat_pos = target[idx[j]]
        similarity_pos = torch.einsum('c,nc->n', [embedding[j], feat_pos.view(-1, dim).detach()])
        logit_pos = torch.div(similarity_pos, temperature)
        logit_pos = logit_pos - max_log.detach()
        exp_logit_pos = torch.exp(logit_pos)
        mask = torch.ones(full_cls * queue, dtype=torch.bool)
        mask[int(idx[j] * queue): int((idx[j] + 1) * queue)] = False
        l_neg = (exp_logit_neg[mask] * dist[mask]).sum().expand(queue)

        # --------------positive sample-------------------
        inter_loss = (-(logit_pos - torch.log((l_neg + exp_logit_pos).clamp(min=1e-4)))).mean()

        intra_loss = torch.div(F.mse_loss(embedding[j].unsqueeze(0), feat_pos.detach()), temperature)
        contrastive_loss += ((inter_loss + 50 * intra_loss))
    contrastive_loss = contrastive_loss / cls
    return contrastive_loss