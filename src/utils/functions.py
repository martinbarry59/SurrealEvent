
import torch
import cv2
def eventstovoxel(events, height=260, width=346, bins=5):
    """
    Converts a batch of events into a voxel grid.
    
    Args:
        events: [B, N, 4] - (t, x, y, p) with t ∈ [0, 1], x ∈ [0, 1], y ∈ [0, 1]
        height, width: spatial resolution
        bins: number of time bins

    Returns:
        voxel: [B, 2 * bins, H, W] voxel grid with separate channels for polarities
    """
    B, N, _ = events.shape
    device = events.device

    # Normalize and quantize to voxel indices
    
    x = (events[:, :, 1] * width).long().clamp(0, width - 1)
    y = (events[:, :, 2] * height).long().clamp(0, height - 1)
    t = (events[:, :, 0] * bins).long().clamp(0, bins - 1)
    
    p = events[:, :, 3].long()
    # Final channel index: [B, N]
    c = t

    voxel = torch.zeros(B, bins, height, width, device=device)
    batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(-1, N)

    voxel.index_put_((batch_idx, c, y, x), p * torch.ones_like(t, dtype=torch.float), accumulate=True)

    return voxel
def eventstohistogram(events, height=260, width=346):
        B, N, _ = events.shape
        x = (events[:, :, 1] * width).long().clamp(0, width - 1)
        y = (events[:, :, 2] * height).long().clamp(0, height - 1)
        p = events[:, :, 3].long().clamp(0, 1)

        hist = torch.zeros(B, 2, height, width, device=events.device)
        batch_idx = torch.arange(B, device=events.device).unsqueeze(1).expand(-1, N)
        hist.index_put_((batch_idx, p, y, x), torch.abs(events[:, :, 3]), accumulate=True)

        return hist

def add_frame_to_video(video_writer, images):
    if images[0].shape[-1] == 4:
        y = torch.round(images[0][0,:,1] * 346)
        x = torch.round(images[0][0,:,2] * 260)
        img = torch.zeros(260, 346).to(images[0].device)
        img[x.long(), y.long()] = 1
    else:
        img = 1 * (torch.sum(images[0][0], dim=0) > 0)
    images[0] = img
    merged = []
    for img in images:
        merged.append(img)
    merged = torch.cat(merged, dim=1).detach().cpu().numpy()
    merged = (merged * 255 ).astype('uint8')
    merged = cv2.cvtColor(merged, cv2.COLOR_GRAY2BGR)  # make it (H, W, 3)
    video_writer.write(merged)  # Write the frame to video

def calc_topk_accuracy(output, target, topk=(1,)):
    """
    Modified from: https://gist.github.com/agermanidis/275b23ad7a10ee89adccf021536bb97e
    Given predicted and ground truth labels,
    calculate top-k accuracies.
    """
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)

    pred = pred.t()
    
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(1 / batch_size))
    return res
