
import torch
import cv2

def apply_event_augmentations(events, training=True, aug_prob=0.5, width=346, height=260):
    """
    Apply various augmentations to event data to improve robustness.
    
    Args:
        events: [B, N, 4] - (t, x, y, p) with t ∈ [0, 1], x ∈ [0, width-1], y ∈ [0, height-1]
        training: whether model is in training mode
        aug_prob: probability of applying augmentations
        width, height: image dimensions for proper coordinate handling
    
    Returns:
        augmented events
    """
    if not training or torch.rand(1).item() > aug_prob:
        return events
    B, N, _ = events.shape
    device = events.device
    augmented = events.clone()
    
    # 1. Temporal jitter (simulate timing noise in real cameras)
    if torch.rand(1).item() < 0.4:
        time_jitter = torch.randn_like(augmented[:, :, 0]) * 0.01  # 1% temporal noise
        augmented[:, :, 0] = (augmented[:, :, 0] + time_jitter).clamp(0, 1)
    
    # 2. Spatial jitter (pixel registration errors)
    if torch.rand(1).item() < 0.3:
        spatial_jitter_x = torch.randn_like(augmented[:, :, 1]) * 1.5  # ~1.5 pixel noise
        spatial_jitter_y = torch.randn_like(augmented[:, :, 2]) * 1.5  # ~1.5 pixel noise
        augmented[:, :, 1] = (augmented[:, :, 1] + spatial_jitter_x).clamp(0, width - 1)
        augmented[:, :, 2] = (augmented[:, :, 2] + spatial_jitter_y).clamp(0, height - 1)
    
    # 3. Event dropout (simulate dead pixels or low-light conditions)
    if torch.rand(1).item() < 0.4:
        dropout_rate = torch.rand(1).item() * 0.1 + 0.05  # 5-15% dropout
        if torch.rand(1).item() < 0.5:
            # Random dropout
            keep_mask = torch.rand(B, N, device=device) > dropout_rate
        else:
            # Spatial cluster dropout (dead pixel regions)
            keep_mask = torch.ones(B, N, dtype=torch.bool, device=device)
            for b in range(B):
                n_clusters = torch.randint(1, 5, (1,)).item()
                for _ in range(n_clusters):
                    # Create spatial clusters of dropped events
                    center_x = torch.rand(1).item() * width  # 0 to width-1
                    center_y = torch.rand(1).item() * height  # 0 to height-1
                    radius = (torch.rand(1).item() * 0.06 + 0.02) * min(width, height)  # 2-8% of image size
                    
                    dist_x = (augmented[b, :, 1] - center_x).abs()
                    dist_y = (augmented[b, :, 2] - center_y).abs()
                    cluster_mask = (dist_x < radius) & (dist_y < radius)
                    keep_mask[b] = keep_mask[b] & ~cluster_mask
        
        # Apply dropout
        new_augmented = []
        for b in range(B):
            batch_events = augmented[b][keep_mask[b]]  # Select events for this batch
            new_augmented.append(batch_events)
        
        # Find the minimum number of events across batches
        min_events = min(batch.shape[0] for batch in new_augmented)
        
        # Pad or truncate to maintain consistent shape
        final_augmented = torch.zeros(B, min(N, min_events), 4, device=device)
        for b in range(B):
            n_events = min(min_events, new_augmented[b].shape[0])
            final_augmented[b, :n_events] = new_augmented[b][:n_events]
        
        # Pad back to original size if needed
        if final_augmented.shape[1] < N:
            padding = torch.zeros(B, N - final_augmented.shape[1], 4, device=device)
            augmented = torch.cat([final_augmented, padding], dim=1)
        else:
            augmented = final_augmented
    
    # 4. Polarity flip (sensor noise)
    if torch.rand(1).item() < 0.2:
        flip_rate = torch.rand(1).item() * 0.04 + 0.01  # 1-5% polarity flips
        flip_mask = torch.rand(B, N, device=device) < flip_rate
        augmented[:, :, 3] = torch.where(flip_mask, -augmented[:, :, 3], augmented[:, :, 3])
    
    # # 5. Temporal stretching/compression (different camera speeds)
    # if torch.rand(1).item() < 0.3:
    #     time_stretch = torch.rand(1).item() * 0.4 + 0.8  # 0.8-1.2
    #     augmented[:, :, 0] = (augmented[:, :, 0] * time_stretch).clamp(0, 1)
    # # 
    # 6. Event rate variation (simulate different lighting conditions)
    if torch.rand(1).item() < 0.25:
        rate_factor = torch.rand(1).item() * 1.0 + 0.5  # 0.5-1.5
        if rate_factor < 1.0:  # Reduce events (low light)
            keep_ratio = rate_factor
            keep_mask = torch.rand(B, N, device=device) < keep_ratio
            
            # Apply rate variation batch-wise
            new_augmented = []
            for b in range(B):
                batch_events = augmented[b][keep_mask[b]]  # Select events for this batch
                new_augmented.append(batch_events)
            
            # Find the minimum number of events across batches
            min_events = min(batch.shape[0] for batch in new_augmented)
            
            # Create final tensor with consistent shape
            final_augmented = torch.zeros(B, min(N, min_events), 4, device=device)
            for b in range(B):
                n_events = min(min_events, new_augmented[b].shape[0])
                final_augmented[b, :n_events] = new_augmented[b][:n_events]
            
            # Pad back to original size if needed
            if final_augmented.shape[1] < N:
                padding = torch.zeros(B, N - final_augmented.shape[1], 4, device=device)
                augmented = torch.cat([final_augmented, padding], dim=1)
            else:
                augmented = final_augmented
    
    return augmented


def add_hot_pixels(events, device, width, height):
    """Add hot pixel events (common in real event cameras)"""
    
    B, N, _ = events.shape
    n_hot_pixels = torch.randint(5, 1000, (1,)).item()
    
    hot_events = torch.zeros(B, n_hot_pixels, 4, device=device)
    hot_events[:, :, 0] = torch.rand(B, n_hot_pixels, device=device)  # Random times
    hot_events[:, :, 1] = torch.rand(B, n_hot_pixels, device=device) * (width - 1)  # Random x (pixel coordinates)
    hot_events[:, :, 2] = torch.rand(B, n_hot_pixels, device=device) * (height - 1)  # Random y (pixel coordinates)
    hot_events[:, :, 3] = torch.randint(0, 2, (B, n_hot_pixels), device=device) * 2 - 1
    
    return torch.cat([events, hot_events], dim=1)


def eventstovoxel(events, height=260, width=346, bins=5, training=True, hotpixel=False, aug_prob=0.5):
    """
    Converts a batch of events into a voxel grid with optional augmentations.
    
    Args:
        events: [B, N, 4] - (t, x, y, p) with t ∈ [0, 1], x ∈ [0, 1], y ∈ [0, 1]
        height, width: spatial resolution
        bins: number of time bins
        training: whether model is in training mode
        aug_prob: probability of applying augmentations during training

    Returns:
        voxel: [B, bins, H, W] voxel grid
    """
    B, N, _ = events.shape
    device = events.device

    # Apply augmentations during training
    if hotpixel:
        events = add_hot_pixels(events, device, width, height)
    if training:
        events = apply_event_augmentations(events, training=training, aug_prob=aug_prob, width=width, height=height)
        B, N, _ = events.shape  # Update shape after augmentations
    # Add hot pixels (realistic camera noise)
    
    # Add random noise events (keeping your existing augmentation)
    
    B, N, _ = events.shape  # Final shape after all augmentations

    # Convert normalized coordinates to pixel indices
    x = (events[:, :, 1]).long().clamp(0, width - 1)
    y = (events[:, :, 2]).long().clamp(0, height - 1)
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
        y = torch.round(images[0][0,:,1])
        x = torch.round(images[0][0,:,2])
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
