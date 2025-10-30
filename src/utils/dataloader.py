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
from utils.functions import  eventstovoxel
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
    def __init__(self, h5_dir, tsne=True, upsampling_factor=12):
        super().__init__()
        self.events_files = glob.glob(os.path.join(h5_dir, "**/*dvs.h5"), recursive = True)
        print(f"Found {len(self.events_files)} event files in {h5_dir}")
        self.depth_files = [f.replace("dvs.h5", "vid_slomo_depth.h5") for f in self.events_files]
        self.tsne = tsne
        self.upsampling_factor = upsampling_factor
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
            events = torch.Tensor(f['vids'][:], )  # shape [N_events, 4]
        with h5py.File(self.depth_files[idx], 'r') as f:
            depth = torch.Tensor(f['vids'][:])  # shape [T, H, W]
        
        ## convert uint8
        depth = remove_border(depth)
        depth = depth.to(torch.uint8)
        depth[depth > 0] = 255
        events = events[:, :4]
        
        # voxels = torch.zeros((depth.shape[0], 5, 260, 346)).to(torch.uint8)
        step = 1/(30*self.upsampling_factor)
        t_events = torch.zeros((depth.shape[0], 10000, 4), requires_grad=False)
        for t in range(depth.shape[0]):
              # [1, 4]
            t_start = max((t - 4) * step, 0)
            ## print all individual event times
               
            nevents = events[ ( t_start <= events[:, 0]) * (events[:, 0] < (t + 1) * step)].unsqueeze(0)
            ## suffle nevents
            nevents = nevents[:, torch.randperm(nevents.shape[1]), :]
            
            if nevents.shape[1] > 10000:
                nevents = nevents[:, :10000, :]
            t_events[t, :nevents.shape[1]] = nevents
        return t_events, depth

        if self.tsne:
            return events, depth, self.events_files[idx].split("/")[-2]  # return the folder name as label
        else:
            return voxels, depth
    # def items_to_cache(self):
    #     for idx in tqdm.tqdm(range(len(self.events_files))):
    #         voxels, depth = self.__getitem__(idx)
    #         ## print types of voxels and depth
    #         print(f"Voxels type: {voxels.dtype}, Depth type: {depth.dtype}")
    #         ## save it in efficient format for later use

    #         cache_dir = self.events_files[idx].replace("dataset", "datasetpt")
    #         if not os.path.exists(cache_dir):
    #             os.makedirs(cache_dir)
    #         cache_file = os.path.join(cache_dir, f"item_{idx}.pt")
    #         torch.save((voxels, depth), cache_file)
    #         return cache_file
def sampling_events(t_old, t_new, events, old_events):
    
    
    # for t in sample[:,0]:
    #     print("{:.30f}".format(t.item()))  # 20 decimal places
    
    
    return events[(events[:, 0] >= t_old )* (events[:, 0] < t_new)]

def Transformer_collate(batch):
    # batch = list of (event_chunks, depth) tuples
    import time 
    start = time.time()
    batched_event_chunks = []
    events = []
    depths = []
    depth_time = time.time()
    for sample in batch:
        depths.append(sample[1])  # [T, H, W]
        events.append(sample[0])  # [T, N_i, 4]
    
    depths = torch.stack(depths).permute((1,0,2,3))  # [T, B, H, W]
    events = torch.stack(events).permute((1, 0,2, 3))  # [T, B, N_i, 4]
    # for _ in range(depths.shape[0]):
    #     # continue
        
    #     t_start = max(t_new - 4 * step, 0)
    #     t_new = t_new + step
    #     # list of [N_i, 4]
        
    #     if len(batch[0]) == 2:
    #         event_histories = [sampling_events(t_start, t_new, events, event_histories[n]) for n, (events, _) in enumerate(batch)]
    #     elif len(batch[0]) == 3:
    #         event_histories = [sampling_events(t_start, t_new, events, event_histories[n]) for n, (events, _, _) in enumerate(batch)]
        
    #     # padded = pad_sequence(event_histories, batch_first=True)  # [B, N_max, 4]
       
    #     # batched_event_chunks.append(padded)
   
    if len(batch[0]) == 2:
        return [events, depths]
    elif len(batch[0]) == 3:
        labels = [sample[2] for sample in batch]
        return [batched_event_chunks, depths, labels]

def CNN_collate(batch):
    
    depths = []
    step = 1/(30 * 12)
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
def get_data(data, t):
    if len(data) == 2:
        events_videos, depths = data
        return events_videos[t], depths[t]
if __name__ == "__main__":
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../dataset/")
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../processed_realvideos/")
    train_dataset = EventDepthDataset(data_path)
    train_dataset.test_corruption()

