import torch
from torch.utils.data import Dataset
import h5py
from torch.nn.utils.rnn import pad_sequence
import os
import glob
import tqdm
from torchvision.transforms import v2 
import shutil
import random
class RandomChoice(torch.nn.Module):
    def __init__(self, transforms):
       super().__init__()
       self.transforms = transforms

    def __call__(self, img):
        t = random.choice(self.transforms)
        return t(img) 
def event_dropout(events, p=0.1):
    mask = torch.rand_like(events) > p
    return events * mask
def remove_border(tensor, edges=1):
    border = torch.zeros_like(tensor)
    ## remove border of last two dimensions without knowing how much dimension the tensor has before
    if len(tensor.shape) == 4:
        border[:, :, edges:-edges, edges:-edges] = 1
    elif len(tensor.shape) == 3:
        border[:, edges:-edges, edges:-edges] = 1
    
    return tensor * border
def apply_augmentations(events, depth):
    ## transfor grayscale images
    transforms = RandomChoice([
        # v2.RandomHorizontalFlip(),
        # v2.RandomVerticalFlip(),
        v2.RandomRotation(180),
    ])
    to_process = torch.cat([events, depth.unsqueeze(2)], dim=2)
    ## border to 0 
    
    
    processed = transforms(to_process)
    events, depth = processed[:, :, :events.shape[2]], processed[:, :, events.shape[2]:].squeeze(2)
    return events, depth
class EventDepthDataset(Dataset):
    def __init__(self, h5_dir, tsne=True):
        super().__init__()
        self.events_files = glob.glob(os.path.join(h5_dir, "**/*dvs.h5"), recursive = True)
        self.depth_files = [f.replace("dvs.h5", "vid_slomo_depth.h5") for f in self.events_files]
        self.tsne = tsne
    def test_corruption(self):
        for i in tqdm.tqdm(range(len(self.events_files))):
            try:
                with h5py.File(self.events_files[i], 'r') as f:
                    events = torch.Tensor(f['vids'][:])  # shape [N_events, 4]
                with h5py.File(self.depth_files[i], 'r') as f:
                    depth = torch.Tensor(f['vids'][:])  # shape [T, H, W]
            except:
                
                print(f"Error reading file {self.events_files[i]} or {self.depth_files[i]}")
                ## delete parent folder
                parent_folder = os.path.dirname(self.events_files[i])
                if os.path.exists(parent_folder):
                    shutil.rmtree(parent_folder)
                    print(f"Deleted folder {parent_folder}")
    def __len__(self):
        return len(self.events_files)

    def __getitem__(self, idx):
        with h5py.File(self.events_files[idx], 'r') as f:
            events = torch.Tensor(f['vids'][:])  # shape [N_events, 4]
        with h5py.File(self.depth_files[idx], 'r') as f:
            depth = torch.Tensor(f['vids'][:])  # shape [T, H, W]

        events = events[:, :4]
        events[:, 1] = events[:, 1] / 346
        events[:, 2] = events[:, 2] / 260
        
        if self.tsne:
            return events, depth, self.events_files[idx].split("/")[-2]  # return the folder name as label
        else:
            return events, depth

def sampling_events(t_old, t_new, events, old_events):
    max_events = 1000
    ## add white noised events
    N_white = torch.randint(0, 10, (1,)).item()
    white_events = torch.zeros((N_white, 4))
    white_events[:, 0] = t_old + torch.rand(N_white) * (t_new - t_old)
    white_events[:, 1] = torch.rand(N_white) * 0.99  # x
    white_events[:, 2] = torch.rand(N_white)* 0.99  # y
    white_events[:, 3] = torch.randint(0, 2, (N_white,)) * 2 - 1  # polaritys
    events = torch.cat([events, white_events], dim=0)
    sample =  events[(events[:, 0] >= t_old )* (events[:, 0] < t_new)]

    
    if sample.shape[0] != 0:
        old_events = torch.cat([old_events, sample], dim=0)
        if len(old_events) > max_events:
            old_events = old_events[-max_events:]
    
    return old_events

def Transformer_collate(batch):
    # batch = list of (event_chunks, depth) tuples
    
    batched_event_chunks = []
    batched_masks = []
    depths = []
    
    for sample in batch:
        depths.append(remove_border(sample[1]) / 255 )  # [T, H, W]

    depths = torch.stack(depths).permute((1,0,2,3))  # [B, T, H, W]
    t_new = 0
    event_histories = [torch.zeros(1,4) for _ in range(depths.shape[0])]
    
    for _ in range(depths.shape[0]):
        # continue
        t_old = t_new
        t_new = t_old + 1/(30*12)
        # list of [N_i, 4]
        
        if len(batch[0]) == 2:
            event_histories = [sampling_events(t_old, t_new, events, event_histories[n]) for n, (events, _) in enumerate(batch)]
        elif len(batch[0]) == 3:
            event_histories = [sampling_events(t_old, t_new, events, event_histories[n]) for n, (events, _, _) in enumerate(batch)]
        
        padded = pad_sequence(event_histories, batch_first=True)  # [B, N_max, 4]
        mask = torch.zeros(padded.shape[:2], dtype=torch.bool)  # [B, N_max]
        for i, ev in enumerate(event_histories):
            mask[i, :ev.size(0)] = 1
        batched_event_chunks.append(padded)
        batched_masks.append(mask)
    if len(batch[0]) == 2:
        return [batched_event_chunks, depths, batched_masks]
    elif len(batch[0]) == 3:
        labels = [sample[2] for sample in batch]
        return [batched_event_chunks, depths, batched_masks, labels]

def CNN_collate(batch):
    
    depths = []
    step = 1/(30*12)
    event_frames = torch.zeros(len(batch), batch[0][1].shape[0], 2 , 260, 346)
    for batch_n,(events, depth) in enumerate(batch):
        events = event_dropout(events, p=0.05)
        depths.append(remove_border(depth) / 255)  # [T, H, W]
        times = events[:, 0]

        x = torch.round(events[:,1] * 346).long()
        y = torch.round(events[:,2] * 260).long()
        polarities = ((events[:, 3] +1)/2).long()
        frame_n = (torch.floor(times / step)-1).long()
        event_frames[batch_n, frame_n, polarities , y, x] = 1
    depths = torch.stack(depths).permute((1,0,2,3))  # [B, T, H, W]
    event_frames = event_frames.permute(1, 0, 2, 3, 4)
    event_frames, depths = apply_augmentations(event_frames, depths)
    return [event_frames, depths]
if __name__ == "__main__":
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../dataset/")
    train_dataset = EventDepthDataset(data_path)
    train_dataset.test_corruption()

def get_data(data, t):
    if len(data) == 2:
        events_videos, depths = data
        return events_videos[t], depths[t], None
    elif len(data) == 3:
        events_videos, depths, masks = data
        events = events_videos[t]
        depth = depths[t]
        mask = masks[t]
        return events, depth, mask
    elif len(data) == 4:
        events_videos, depths, masks, labels = data
        events = events_videos[t]
        depth = depths[t]
        mask = masks[t]
        label = [str_label+"_t_"+ str(t) for str_label in labels]
        return events, depth, mask, label
    else:
        raise ValueError("Data must be a tuple of length 2, 3, or 4.")