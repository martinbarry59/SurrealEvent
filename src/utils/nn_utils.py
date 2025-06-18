import random

from sklearn.manifold import TSNE
from utils.functions import calc_topk_accuracy
from utils.plot_utils import plotly_TSNE
from utils.dataloader import get_data
import torch
import numpy as np
import torch.nn.functional as F
from models.TransformerEventSurreal import EventTransformer
from models.BOBVEG import BestOfBothWorld
from utils.dataloader import EventDepthDataset, CNN_collate, Transformer_collate

from config import data_path, results_path, checkpoint_path
import os
import shutil

def variance_loss(x, eps=1e-4):
    # x: [B, D]
    std = torch.sqrt(x.var(dim=0) + eps)
    return torch.mean(F.relu(1.0 - std))
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

def edge_aware_loss(pred, target):
    
    pred_dx = pred[ :, :, 1:] - pred[ :, :, :-1]
    target_dx = target[:, :, 1:] - target[:, :, :-1]
    pred_dy = pred[ :, 1:, :] - pred[ :, :-1, :]
    target_dy = target[ :, 1:, :] - target[ :, :-1, :]
    return F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)



def process_output(mask):
    """task mask as input, compute the target for contrastive loss"""
    (T1, B1, T2, B2) = mask.size()  # [B, P, SQ, B, N, SQ]
    target = mask == 1
    target = target * 1
    target.requires_grad = False
    return target, (B1, T1, B2, T2)

def get_score(predictions, GT):
    if predictions.dim() == 5:
        T1, B, C, H, W = predictions.shape
        predictions = predictions.view(T1*B, C * H * W)
        T2, B, C, H, W = GT.shape
        GT = GT.view(T2*B, C * H * W).transpose(0, 1)
    else:
        T1, B, C = predictions.shape
        T2, B, C = GT.shape
        predictions = predictions.view(T1*B, C)
        GT = GT.view(T2*B, C).transpose(0, 1)
    pred_norm = torch.nn.functional.normalize(predictions, dim=1)
    GT_norm = torch.nn.functional.normalize(GT, dim=1)
    
    score = torch.matmul(pred_norm, GT_norm) / 0.1
    return score.view(T1,B,T2, B)
def compute_CPC_loss(predictions, GT, criterion, mask = True):
    if predictions.dim() == 5:
        T1, B1, C, H, W = predictions.shape
        T2, B2, C, H, W = GT.shape
    else:
        T1, B1, C = predictions.shape
        T2, B2, C = GT.shape
    score = get_score(predictions, GT)
    
    mask_ = (
                    torch.zeros((T1, B1, T2, B2), dtype=torch.int8, requires_grad=False, device=score.device)
                    .detach()
                )
    if mask:
        mask_[
            torch.arange(T1, device=mask_.device),
            torch.arange(B1, device=mask_.device)[:, None],
            torch.arange(T2, device=mask_.device),
            torch.arange(B2, device=mask_.device)[:, None]
        ] = 1
    target_, (B1, T1, B2, T2) = process_output(mask_)
    score_flattened = score.view(B1 * T1, B2 * T2)
    target_flattened = target_.contiguous().view(B1 * T1, B2 * T2)
    

    
    return target_flattened, score_flattened

def vicreg_loss(z, z_pos, sim_coeff=25, var_coeff=25, cov_coeff=1, eps=1e-4, gamma=1.0):
    # Invariance
    z = z.view(-1, z.size(-1))
    z_pos = z_pos.view(-1, z_pos.size(-1))
    inv_loss = F.mse_loss(z, z_pos)

    # Variance
    std_z = torch.sqrt(z.var(dim=0) + eps)
    std_z_pos = torch.sqrt(z_pos.var(dim=0) + eps)
    var_loss = torch.mean(F.relu(gamma - std_z)) + torch.mean(F.relu(gamma - std_z_pos))

    # Covariance
    z_centered = z - z.mean(dim=0)
    cov_z = (z_centered.T @ z_centered) / (z.size(0) - 1)
    off_diag_z = cov_z.flatten()[::cov_z.size(1)+1]  # skipping diag
    cov_loss = off_diag_z.pow(2).sum() / z.size(1)
    # Same for z_pos
    z_pos_centered = z_pos - z_pos.mean(dim=0)
    cov_zp = (z_pos_centered.T @ z_pos_centered) / (z_pos.size(0) - 1)
    off_diag_zp = cov_zp.flatten()[::cov_zp.size(1)+1]
    cov_loss = cov_loss + off_diag_zp.pow(2).sum() / z_pos.size(1)

    return sim_coeff * inv_loss + var_coeff * var_loss + cov_coeff * cov_loss


def forward_feed(model, data, device, step_size=1, start_seq=0, block_update=30):
    seq_events = []
    seq_masks = []
    seq_depths = []
    seq_labels = []
    for t in range(start_seq, start_seq + block_update * step_size, step_size):   
        datat = get_data(data, t)
        events, depth , mask = datat[:3]
        events, mask = events.to(device), mask.to(device), depth.to(device)
        
        labels = None
        if len(datat) == 4:
            labels = datat[3]
            
       
        seq_events.append(events)
        seq_masks.append(mask)
        seq_depths.append(depth)
        if labels is not None:
            seq_labels.append(labels)
    ## convert the seq_labels to numpy array
    if len(seq_labels) > 0:
        seq_labels = np.array(seq_labels)
    predictions, encodings = model(seq_events, seq_masks)
    return predictions, encodings, seq_labels, seq_depths


def compute_loss(predictions, encodings, criterion):
    predictions = predictions.permute(1,0,2).contiguous()[:-1]
    encodings = encodings.permute(1,0,2).contiguous()[1:]
    
    target, score = compute_CPC_loss(predictions, encodings, criterion)
    target = target.argmax(dim=1)
    score /= 0.07
    loss = criterion(score, target)

    loss += vicreg_loss(predictions, encodings)
    top1, top3, top5 = calc_topk_accuracy(score, target, (1, 3, 5))
    return loss, top1, top3, top5, score, target
def update(loss, model, optimizer, scaler):
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()

def sequence_for_LSTM(data, model, criterion, optimizer, device,
                    train = True, scaler = None,
                    len_videos=1188, training_steps=300, block_update=30, rand_step=True
                    ):
    if rand_step:
        step_size =  torch.randint(1, 10, (1,)).item()
    else:
        ## fixed step size
        step_size = 1

    N_update = int(training_steps / block_update)
    t_start = random.randint(10, len_videos - N_update * block_update)
    loss_avg = []
    for n in range(N_update):
        
        start_seq = t_start + n * block_update * step_size
        if (start_seq + block_update * step_size) > len_videos - 1:
            break
        predictions, encodings, labels, depths = forward_feed(model, data, device, step_size=step_size, start_seq=start_seq, block_update=block_update)
        
        if n == 0 and len(labels) > 0:
            with torch.no_grad():
                labels = np.permute_dims(labels, (1, 0))
                ## flatten labels
                labels = labels.reshape(-1)
                video_labels = labels.copy()
                video_labels = np.array([ l.split("_t_")[0] for l in video_labels])
                ## repeat to all other dimensions
                ## flatten encodings
                flattened_encodings = encodings.reshape(-1, encodings.shape[-1])
                # print(encodings.shape, predictions.shape, flattened_encodings.shape, labels.shape, unique_labels.shape)
                tsne = TSNE(n_components=2, verbose=0, perplexity=50, max_iter=2000).fit_transform(
                    flattened_encodings.cpu().numpy()
                )
                plotly_TSNE(tsne, labels, video_labels, f"{model.model_type}")
        # MSE_loss = F.mse_loss(outputs, depth.unsqueeze(1).repeat(1, 3, 1, 1))
        ## compute SSIM loss
        # SSIM_loss = torch.torchev
        loss = criterion(predictions, depths)
        #     loss_t = F.l1_loss(previous_output, outputs)
        #     loss_est = torch.exp(- 50 * torch.nn.MSELoss()(depth_previous, depth))
        #     TC_loss = loss_t * loss_est
        #     loss += 50 * TC_loss / block_update
        with torch.no_grad():
                    
            if video_writer:
                
                if kerneled is not None:
                    kerneled = kerneled-kerneled.min() 
                    kerneled = kerneled/(kerneled.max()+1e-6)
                    images_to_write = [events, kerneled[0,0], kerneled[0,1], depth[0], outputs[0,0].squeeze(0)]
                else:
                    images_to_write = [events, depth[0], outputs[0,0].squeeze(0)]
                add_frame_to_video(video_writer, images_to_write)
        loss_avg.append(loss.item())
        if train:
            update(loss, model, optimizer, scaler)
            
        # if n == 0:
        #     check_CPC_representation_collapse(predictions, encodings)
    return predictions, loss_avg
def sequence_for_LSTM_CPC(data, model, criterion, optimizer, device,
                    train = True, scaler = None,
                    len_videos=1188, training_steps=300, block_update=30, rand_step=True
                    ):
    if rand_step:
        step_size =  torch.randint(1, 10, (1,)).item()
    else:
        ## fixed step size
        step_size = 1

    N_update = int(training_steps / block_update)
    t_start = random.randint(10, len_videos - N_update * block_update)
    t_end = t_start + N_update * block_update
    loss_avg = []
    top1_avg, top3_avg, top5_avg = [], [], []
    for n in range(N_update):
        
        start_seq = t_start + n * block_update * step_size
        if (start_seq + block_update * step_size) > len_videos - 1:
            break
        predictions, encodings, labels = forward_feed(model, data, device, step_size=step_size, start_seq=start_seq, block_update=block_update)
        
        if n == 0 and len(labels) > 0:
            with torch.no_grad():
                labels = np.permute_dims(labels, (1, 0))
                ## flatten labels
                labels = labels.reshape(-1)
                video_labels = labels.copy()
                video_labels = np.array([ l.split("_t_")[0] for l in video_labels])
                ## repeat to all other dimensions
                ## flatten encodings
                flattened_encodings = encodings.reshape(-1, encodings.shape[-1])
                # print(encodings.shape, predictions.shape, flattened_encodings.shape, labels.shape, unique_labels.shape)
                tsne = TSNE(n_components=2, verbose=0, perplexity=50, max_iter=2000).fit_transform(
                    flattened_encodings.cpu().numpy()
                )
                plotly_TSNE(tsne, labels, video_labels, f"{model.model_type}")
        
        loss, top1, top3, top5, score, target = compute_loss(predictions, encodings, criterion)
        loss_avg.append(loss.item())
        top1_avg.append(top1.item())
        top3_avg.append(top3.item())
        top5_avg.append(top5.item())
        del score, target
        if train:
            update(loss, model, optimizer, scaler)
            
        # if n == 0:
        #     check_CPC_representation_collapse(predictions, encodings)
    return loss_avg, top1_avg, top3_avg, top5_avg


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