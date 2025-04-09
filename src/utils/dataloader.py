import torch
from torch.utils.data import Dataset
import h5py
from torch.nn.utils.rnn import pad_sequence
import os
import glob
class EventDepthDataset(Dataset):
    def __init__(self, h5_dir):
        super().__init__()
        self.events_files = glob.glob(os.path.join(h5_dir, "**/*dvs.h5"), recursive = True)
        self.depth_files = [f.replace("dvs.h5", "vid_slomo_depth.h5") for f in self.events_files]

    def __len__(self):
        return len(self.events_files)

    def __getitem__(self, idx):
        with h5py.File(self.events_files[idx], 'r') as f:
            events = torch.Tensor(f['vids'][:])  # shape [N_events, 4]
        with h5py.File(self.depth_files[idx], 'r') as f:
            depth = torch.Tensor(f['vids'][:])  # shape [T, H, W]

        

        return events, depth

def sampling_events(t_old, t_new, events):
    
    sample =  events[(events[:, 0] >= t_old )* (events[:, 0] < t_new)][:,:-1]
    ## normalise x ,y 
    sample[:,1] = sample[:,1]/346
    sample[:,2] = sample[:,2]/260
    if sample.shape[0] == 0:
        sample = torch.zeros(1, 4)
    return sample

def collate_event_batches(batch):
    # batch = list of (event_chunks, depth) tuples
    batched_event_chunks = []
    # batched_masks = []
    depths = []
    for sample in batch:
        depths.append(sample[1] / 255 )  # [T, H, W]

    depths = torch.stack(depths).permute((1,0,2,3))  # [B, T, H, W]
    t_new = 0
    for frame_step in range(depths.shape[0]):
        t_old = t_new
        t_new = t_old + 1/(30*12)
         # list of [N_i, 4]
        batch_events = [sampling_events(t_old, t_new, events) for events, _ in batch]
        
        padded = pad_sequence(batch_events, batch_first=True)  # [B, N_max, 4]
        # mask = torch.zeros(padded.shape[:2], dtype=torch.bool)  # [B, N_max]
        # for i, ev in enumerate(batch_events):
        #     mask[i, :ev.size(0)] = 1
        batched_event_chunks.append(padded)
        # batched_masks.append(mask)
        

    
    return batched_event_chunks, depths

# Example usage:
# dataset = EventDepthDataset('/path/to/h5/data')
# loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_event_batches)
