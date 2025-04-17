import h5py
from config import data_path
from utils.functions import eventstohistogram
from utils.dataloader import EventDepthDataset
import torch
from utils.dataloader import collate_event_batches
import matplotlib.pyplot as plt
import cv2
if __name__ == "__main__":
    file_path = f"{data_path}/train/run0/01_01/01_01_c0001/"
    
    gray_scale = h5py.File(f"{file_path}/vid_slomo.h5", "r")
    train_dataset = EventDepthDataset(file_path)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collate_event_batches)
    for batch in train_loader:
        events, depth = batch
        break
    step = 100
    
    print(depth.shape)
    print(len(events))
    events = events[step] 
    depth = depth[0].squeeze(0)
    event_img   = eventstohistogram(events).squeeze(0).sum(dim=0)
    gray_scale = gray_scale["vids"][:]
    gray_scale = torch.Tensor(gray_scale)[step]
    print(depth.max(), gray_scale.max())
    
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[1].imshow(depth.cpu().numpy(), cmap='gray')
    ax[1].set_title("Depth")
    ax[1].axis('off')
    ax[0].axis('off')
    ax[0].imshow(gray_scale, cmap='gray')
    ax[0].set_title("Original Gray Scale")
    ax[2].imshow(event_img.cpu().numpy(), cmap='gray')
    ax[2].set_title("Events")
    ax[2].axis('off')
    # increase font size
    # increase title font size
    for a in ax:
        a.title.set_fontsize(20)
    plt.tight_layout()
    plt.savefig(f"event_gray_depth.pdf")
    plt.show()
