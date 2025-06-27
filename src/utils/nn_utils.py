import random
from utils.functions import add_frame_to_video
from sklearn.manifold import TSNE
from utils.plot_utils import plotly_TSNE
from utils.dataloader import EventDepthDataset, CNN_collate, Transformer_collate, get_data
import torch
import numpy as np
import tqdm
import torch.nn.functional as F
from models.TransformerEventSurreal import EventTransformer
from models.BOBVEG import BestOfBothWorld
from config import data_path, results_path, checkpoint_path
import os
import shutil
from ignite.metrics import SSIM
from ignite.engine import Engine
def eval_step(engine, batch):
        return batch
default_evaluator = Engine(eval_step)
SSIM_metric = SSIM(data_range=1.0)
SSIM_metric.attach(default_evaluator, 'ssim')

# def plot_attention_map(attn_weights, events, b, q, img):
#     query_attention = attn_weights[b, q]  # shape: [N]
#     coords = events[b, :, 1:3].cpu().numpy()  # normalized x, y
#     xs = coords[:, 0] * img.shape[1]
#     ys = coords[:, 1] * img.shape[0]
#     plt.imshow(img.cpu().numpy(), cmap='gray')
#     plt.axis('off')
#     plt.scatter(xs, ys, c=query_attention.cpu().numpy(), cmap='viridis', s=5)
#     plt.colorbar(label='Attention Weight')
#     plt.title(f"Query {q} attention over events")
#     plt.show()



def process_output(mask):
    """task mask as input, compute the target for contrastive loss"""
    (T1, B1, T2, B2) = mask.size()  # [B, P, SQ, B, N, SQ]
    target = mask == 1
    target = target * 1
    target.requires_grad = False
    return target, (B1, T1, B2, T2)




def forward_feed(model, data, device, step_size=1, start_seq=0, block_update=30, video_writer=None):
    seq_events = []
    seq_masks = []
    seq_depths = []
    seq_labels = []
    max_t = start_seq + block_update * step_size if block_update > 0 else len(data[0]) - 1
    for t in range(start_seq, max_t, step_size):   
        datat = get_data(data, t)
        events, depth , mask = datat[:3]
        events, mask, depth = events.to(device), mask.to(device), depth.to(device)
        
        labels = None

        if len(datat) == 4:
            labels = datat[3]
            
       
        seq_events.append(events)
        seq_masks.append(mask)
        seq_depths.append(depth)
        if labels is not None:
            seq_labels.append(labels)
    ## convert the seq_labels to numpy array
    seq_depths = torch.stack(seq_depths, dim=1)
    if len(seq_labels) > 0:
        seq_labels = np.array(seq_labels)
    predictions, encodings = model(seq_events, seq_masks)
    with torch.no_grad():
        print(video_writer)        
        if video_writer:
            # for t in range(predictions.shape[1]):
            for t in range(predictions.shape[1]):
                events = seq_events[t]
                depth = seq_depths[:,t]
                outputs = predictions[:,t]
                images_to_write = [events, depth[0], outputs[0]]
                add_frame_to_video(video_writer, images_to_write)
    return predictions, encodings, seq_labels, seq_depths



def compute_LPIPS_loss(predictions, depths, criterion):
    loss = 0
    predictions.unsqueeze_(2)
    predictions = predictions.repeat(1, 1, 3, 1, 1)
    depths.unsqueeze_(2)
    depths = depths.repeat(1, 1, 3, 1, 1)
    for t in range(predictions.shape[1]):
        pred = predictions[:,t]
        enc = depths[:,t]
        loss += criterion(pred, enc).mean()
        if t > 0:
            loss_est = torch.exp(-50 * torch.nn.MSELoss()(depths[:,t], depths[:,t-1]))
            loss_t = F.l1_loss(predictions[:,t], predictions[:,t-1])
            TC_loss = loss_t * loss_est
            loss += 50 * TC_loss / predictions.shape[1]
    return loss / predictions.shape[1]


def update(model, optimizer, scaler):
    
    scaler.unscale_(optimizer)
    # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()
    


def sequence_for_LSTM(data, model, criterion, optimizer, device,
                    train = True, scaler = None,
                    len_videos=1188, training_steps=300, block_update=25, rand_step=True,
                    video_writer=None):
    if rand_step:
        step_size =  torch.randint(1, 10, (1,)).item()
    else:
        ## fixed step size
        step_size = 1
    if train:
        N_update = int(training_steps / block_update)
        t_start = random.randint(10, len_videos - N_update * block_update)
    else:
        N_update = 1
        t_start = 10
        block_update = -1
    loss_avg = []
    loss_MSE = []
    loss_SSIM = []
    # print(f"Starting training from {t_start} for {N_update} updates with block size {block_update} and step size {step_size}")
    optimizer.zero_grad()
    for n in range(N_update):
        
        start_seq = t_start + n * block_update * step_size
        if (start_seq + block_update * step_size) > len_videos - 1:
            break
        predictions, encodings, labels, depths = forward_feed(model, data, device, step_size=step_size, 
                                                              start_seq=start_seq, block_update=block_update, 
                                                              video_writer=video_writer)
        
        # if n == 0 and len(labels) > 0:
        #     with torch.no_grad():
        #         labels = np.permute_dims(labels, (1, 0))
        #         ## flatten labels
        #         labels = labels.reshape(-1)
        #         video_labels = labels.copy()
        #         video_labels = np.array([ l.split("_t_")[0] for l in video_labels])
        #         ## repeat to all other dimensions
        #         ## flatten encodings
        #         flattened_encodings = encodings.reshape(-1, encodings.shape[-1])
        #         # print(encodings.shape, predictions.shape, flattened_encodings.shape, labels.shape, unique_labels.shape)
        #         tsne = TSNE(n_components=2, verbose=0, perplexity=50, max_iter=2000).fit_transform(
        #             flattened_encodings.cpu().numpy()
        #         )
        #         plotly_TSNE(tsne, labels, video_labels, f"{model.model_type}")
        
        
        ## compute SSIM loss
        # SSIM_loss = torch.torchev
        loss = compute_LPIPS_loss(predictions, depths, criterion)
        # if not train:
        if train:
            scaler.scale(loss).backward()
        model.detach_states()
        loss_avg.append(loss.item())
        del loss, encodings
        with torch.no_grad():
            predictions = predictions.view(-1, 1, predictions.shape[-2], predictions.shape[-1]).detach()
            depths = depths.view(-1, 1, depths.shape[-2], depths.shape[-1]).detach()
            MSE = F.mse_loss(predictions, depths)
            loss_MSE.append(MSE.item())
            ## send depths to same type as predictions
            depths = depths.type_as(predictions)
            state = default_evaluator.run([[predictions, depths]])
            loss_SSIM.append(state.metrics['ssim'])  
        
        
    if train:
        update(model, optimizer, scaler)
            
        # if n == 0:
        #     check_CPC_representation_collapse(predictions, encodings)
    model.reset_states()
    return loss_avg, loss_MSE, loss_SSIM

def init_simulation(device, batch_size=10, network="Transformer", checkpoint_file = None):
    

    loading_method = CNN_collate if (not ((network =="Transformer") or ("BOBW" in network))) else Transformer_collate
    train_dataset = EventDepthDataset(data_path+"/train/")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=loading_method)
    test_dataset = EventDepthDataset(data_path+"/test/")
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=loading_method)
    epoch_checkpoint = 0

    # Load the model UNetMobileNetSurreal
    
    save_path = f'{results_path}/{network}'
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path)
    if network == "Transformer":
        model = EventTransformer()
    elif "BOBW" in network:
        model = BestOfBothWorld(model_type=network)
    # else:
    #     model = UNetMobileNetSurreal(in_channels = 2, out_channels = 1, net_type = network , method = method) ## if no LSTM use there we give previous output as input
    if checkpoint_path:

        checkpoint_file = f'{checkpoint_path}/{checkpoint_file}'
        
        print(f"Loading checkpoint from {checkpoint_file}")
        try:
            model.load_state_dict(torch.load(checkpoint_file, map_location=device))
            epoch_checkpoint = int(checkpoint_file.split("_")[2]) + 1
        except:
            print("Checkpoint not found, starting from scratch")
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    criterion.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)  # 10 = total number of epochs
    return model, train_loader, test_loader, optimizer, scheduler, criterion, save_path, epoch_checkpoint