import torch

def to_one_hot_tensor(seg, all_seg_labels):
    B, C, D, W, H = seg.shape
    result = torch.zeros(size=(B, len(all_seg_labels), D, W, H), device=seg.device).to(torch.float32)
    for i, l in enumerate(all_seg_labels):
        result[:, i, :, :, :][seg.squeeze(1) == l] = 1
    return result