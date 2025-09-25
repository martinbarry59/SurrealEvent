import random
from utils.functions import add_frame_to_video
from utils.dataloader import EventDepthDataset, CNN_collate, Transformer_collate, get_data
import torch
import numpy as np
import torch.nn.functional as F
from models.TransformerEventSurreal import EventTransformer
from config import data_path, results_path, checkpoint_path
import os
import shutil
from utils.functions import create_persistent_noise_generator_with_augmentations

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


def compute_edge_loss(pred, target):
    """Compute edge-preserving loss using Sobel filters"""
    # Sobel filters for edge detection
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=pred.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=pred.device).view(1, 1, 3, 3)
    
    # Compute edges
    pred_edge_x = F.conv2d(pred, sobel_x, padding=1)
    pred_edge_y = F.conv2d(pred, sobel_y, padding=1)
    target_edge_x = F.conv2d(target, sobel_x, padding=1)
    target_edge_y = F.conv2d(target, sobel_y, padding=1)
    
    # Edge magnitude
    eps = 1e-8
    pred_edges = torch.sqrt(pred_edge_x**2 + pred_edge_y**2 + eps)
    target_edges = torch.sqrt(target_edge_x**2 + target_edge_y**2 + eps)
    
    return F.l1_loss(pred_edges, target_edges)
    
def process_output(mask):
    """task mask as input, compute the target for contrastive loss"""
    (T1, B1, T2, B2) = mask.size()  # [B, P, SQ, B, N, SQ]
    target = mask == 1
    target = target * 1
    target.requires_grad = False
    return target, (B1, T1, B2, T2)




def forward_feed(model, data, device, train, step_size=1, start_seq=0, block_update=30, video_writer=None, zeroing=False, hotpixel=False, noise_gen=None, zero_all=False):
    seq_events = []
    seq_depths = []
    seq_labels = []
    max_t = start_seq.max().item() + block_update * step_size if block_update > 0 else len(data[0]) - 1
    with torch.no_grad():
        t_entry = start_seq
        for t in range(start_seq.max().item(), max_t, step_size):  
            t_entry[not zeroing] = t
            datat = get_data(data, t_entry, step_size)
            events, depth = datat[:2]
            ## add white noise (-1 or 1 ) with 10% probability
            events, depth = events, depth
            # events = events if not zeroing else events * 0
            events = events * (1-zero_all.to(torch.uint8)).view(-1,1,1) * (1-zeroing.to(torch.uint8)).view(-1,1,1)  # apply zero_all mask
            depth = depth * (1-zero_all.to(torch.uint8)).view(-1,1,1)  # apply zero_all mask

            events = events.to(device)
            depth = depth.to(device)
            t_min, t_max = events[:,:,0].min().item(), events[:,:,0].max().item()
            noise_events = noise_gen.step(events.shape[0], t_min, t_max) 
            if noise_events is not None and noise_events.shape[0] > 0:
                events = torch.cat((events, noise_events), dim=1)
            labels = None

            if len(datat) == 4:
                labels = datat[3]
            seq_events.append(events.to(torch.float32))
            seq_depths.append(depth / 255)
            if labels is not None:
                seq_labels.append(labels)
        ## convert the seq_labels to numpy array
        seq_depths = torch.stack(seq_depths, dim=1)
        if len(seq_labels) > 0:
            seq_labels = np.array(seq_labels)

    predictions, encodings, seq_events = model(seq_events, training=train, hotpixel=hotpixel)
    with torch.no_grad():
        if video_writer:
            # for t in range(predictions.shape[1]):
            for t in range(predictions.shape[1]):
                events = seq_events[t]
                depth = seq_depths[:,t]
                outputs = predictions[:,t]
                images_to_write = [events, depth[0], outputs[0]]
                add_frame_to_video(video_writer, images_to_write)
    return predictions, encodings, seq_labels, seq_depths



def compute_mixed_loss(predictions, depths, criterion, epoch):
    loss = 0
    predictions = predictions.unsqueeze(2)
    predictions = predictions.repeat(1, 1, 3, 1, 1)
    depths = depths.unsqueeze(2)
    depths = depths.repeat(1, 1, 3, 1, 1)
    for t in range(predictions.shape[1]):
        pred = predictions[:,t]
        enc = depths[:,t]
        pred_lpips = pred * 2 - 1
        enc_lpips = enc * 2 - 1
        loss += criterion(pred_lpips, enc_lpips).mean()
        if epoch > 0:
            loss += min(1, epoch) * compute_edge_loss(pred[:,0:1], enc[:,0:1])
            if t > 0:
                mse = torch.nn.MSELoss()(depths[:,t], depths[:,t-1])
                loss_est = torch.exp(torch.clamp(-50 * mse, min=-10, max=10))
                loss_t = F.l1_loss(predictions[:,t], predictions[:,t-1])
                TC_loss = loss_t * loss_est
                
                loss += 1 * min(1, max(0, (epoch)/3) * TC_loss)
    return loss / predictions.shape[1]


def update(model, optimizer, scaler):
    
    scaler.unscale_(optimizer)
    # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()


def sequence_for_LSTM(data, model, criterion, optimizer, device,
                    train = True,  epoch = 0, scaler = None,
                    len_videos=1188, training_steps=300, block_update=25, rand_step=True,
                    video_writer=None):
    step_size =  torch.randint(5, 40, (1,)).item() if train else 5

    if train:
        N_update = int(len_videos / (block_update* step_size))
        t_start = random.randint(0, len_videos - N_update * block_update * step_size - 1)
    else:
        training_steps = len_videos
        N_update = int(training_steps / block_update) - 1
        t_start = 0
    loss_avg = []
    loss_MSE = []
    loss_SSIM = []
    # print(f"Starting training from {t_start} for {N_update} updates with block size {block_update} and step size {step_size}")
    optimizer.zero_grad()
    zero_run = torch.rand(data[0].shape[1]) < .1 if train else torch.ones(data[0].shape[1])
    zero_all = torch.rand(data[0].shape[1]) < 0.01 if train else torch.zeros(data[0].shape[1])
    hotpixel = True if torch.rand(1).item() < 0.3 else False
    config = random.choice([None, 'minimal', 'nighttime']) if train else 'minimal'
    noise_gen = create_persistent_noise_generator_with_augmentations(width=346, height=260, device=device, config_type=config, training=True, seed=None)
    noise_gen.reset()  # per video
    for n in range(N_update):
        steps = n * torch.ones_like(zero_run).long()
        if n>2:
            zeroing = zero_run.bool()
            steps[zeroing] = 3
        else:
            zeroing = torch.zeros_like(zero_run).bool()
        
        start_seq = t_start + steps * block_update * step_size 
        if (start_seq.max().item() + block_update * step_size) > len_videos - 1:
            break
        predictions, encodings, labels, depths = forward_feed(model, data, device, train, step_size=step_size, 
                                                              start_seq=start_seq, block_update=block_update, 
                                                              video_writer=video_writer, zeroing=zeroing, hotpixel=hotpixel, noise_gen=noise_gen, zero_all=zero_all)

        
        
        ## compute SSIM loss
        loss = compute_mixed_loss(predictions, depths, criterion, epoch)

        predictions = predictions.detach()
        depths = depths.detach()
        # if not train:
        if train:
            scaler.scale(loss).backward()
            # total_norm = 0
            # for p in model.parameters():
            #     if p.grad is not None:
            #         param_norm = p.grad.data.norm(2)
            #         print(f"{p.shape} grad norm: {param_norm:.6f}")
            # exit()
        model.detach_states()
        loss_avg.append(loss.item())
        del loss
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
    return loss_avg, loss_MSE, loss_SSIM, step_size, zero_run.float().mean().item(), zero_all.float().mean().item()

