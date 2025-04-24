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
    def __init__(self, h5_dir):
        super().__init__()
        self.events_files = glob.glob(os.path.join(h5_dir, "**/*dvs.h5"), recursive = True)
        print(h5_dir)
        print(len(self.events_files))
        self.depth_files = [f.replace("dvs.h5", "vid_slomo_depth.h5") for f in self.events_files]
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
        return events, depth
        ## repeat events for each time step
        # binned = events[:, 0] / (1/(30*12))
        # binned = torch.floor(binned).long()
        # max_events = 1000
        # event_list = torch.zeros(depth.shape[0], max_events, 4)
        # active_mask = torch.zeros(depth.shape[0], max_events, dtype=torch.bool)
        # active_mask[:, 0] = 1
        # for time in range(depth.shape[0]):
        #     bin_idx = torch.argmax(1*(binned > time))
            
        #     min_idx = max(bin_idx - max_events,0)
        #     tmp = events[min_idx:bin_idx, :]
        #     event_list[time,:min(bin_idx,max_events)] = tmp
        #     active_mask[time, :min(bin_idx,max_events)] = 1
        # return event_list, remove_border(depth) / 255, active_mask


def sampling_events(t_old, t_new, events, old_events):
    max_events = 1000
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
        event_histories = [sampling_events(t_old, t_new, events, event_histories[n]) for n, (events, _) in enumerate(batch)]
        
        padded = pad_sequence(event_histories, batch_first=True)  # [B, N_max, 4]
        mask = torch.zeros(padded.shape[:2], dtype=torch.bool)  # [B, N_max]
        for i, ev in enumerate(event_histories):
            mask[i, :ev.size(0)] = 1
        batched_event_chunks.append(padded)
        batched_masks.append(mask)
    return [batched_event_chunks, depths, batched_masks]

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
    event_frames = event_frames.permute(1,0, 2, 3, 4)
    event_frames, depths = apply_augmentations(event_frames, depths)
    return [event_frames, depths]
if __name__ == "__main__":
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../dataset/")
    train_dataset = EventDepthDataset(data_path)
    train_dataset.test_corruption()