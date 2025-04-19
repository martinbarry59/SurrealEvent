
import torch
import cv2
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