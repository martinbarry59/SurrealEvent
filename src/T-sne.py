# -*- coding: utf-8 -*-
# @Time    : 2022/6/10 14:18
# @Author  : Martin Barry

from __future__ import print_function
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
## add parent folder to path
from models.MobileNetSurreal import UNetMobileNetSurreal
from models.TransformerEventSurreal import EventTransformer
from models.BOBW_CPC import BestOfBothWorld
from utils.dataloader import EventDepthDataset, CNN_collate, Transformer_collate
from utils.functions import add_frame_to_video, calc_topk_accuracy
import torch
from config import data_path, results_path, checkpoint_path
import cv2
import tqdm
import os
from sklearn.manifold import TSNE
import plotly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or use 'XVID' for .avi
from torch.amp import autocast
import shutil
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
                    torch.zeros((T1, B1, T2, B2), dtype=torch.int8, requires_grad=False)
                    .detach()
                    .cuda()
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
    
def get_data(data, t):
    if len(data) == 2:
        events_videos, depths = data
        return events_videos[t].to(device), depths[t].to(device), None
    elif len(data) == 3:
        events_videos, depths, masks = data
        events = events_videos[t].to(device)
        depth = depths[t].to(device)
        mask = masks[t].to(device)
        return events, depth, mask
    else:
        raise ValueError("Data must be a tuple of length 2 or 3")
def main():
    batch_size = 28
    network = "BOBWLSTM_CPC" # LSTM, Transformer, BOBWFF, BOBWLSTM
    

    loading_method = CNN_collate if (not ((network =="Transformer") or ("BOBW" in network))) else Transformer_collate
    train_dataset = EventDepthDataset(data_path+"/train/")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=loading_method)
   
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
        checkpoint_file = f'{checkpoint_path}/model_epoch_7_{model.model_type}_{model.method}.pth'
        
        print(f"Loading checkpoint from {checkpoint_file}")
        try:
            model.load_state_dict(torch.load(checkpoint_file))
            epoch_checkpoint = int(checkpoint_file.split("_")[2]) + 1
        except:
            print("Checkpoint not found, starting from scratch")
    model.to(device)

    # criterion = torch.nn.SmoothL1Loss()
    # criterion = lpips.LPIPS(net='alex')
    criterion = torch.nn.CrossEntropyLoss()
    criterion.to(device)
    run_TSNE(train_loader, model)


def process_output(mask):
    """task mask as input, compute the target for contrastive loss"""
    (B1, T1, B2, T2) = mask.size()  # [B, P, SQ, B, N, SQ]
    target = mask == 1
    target = target * 1
    target.requires_grad = False
    return target, (B1, T1, B2, T2)


# def re_order_labels(labels):
#     ## sort and keep index
#     ## get unique labels
#     unique_labels = torch.unique(labels)

#     ## get the number of unique labels
#     num_labels = unique_labels.size(0)
#     ## create a new label list
#     new_labels = torch.zeros_like(labels)
#     ## loop through the unique labels
#     for i in range(num_labels):
#         ## get the index of the unique label
#         idx_label = labels == unique_labels[i]
#         ## assign new labels
#         new_labels[idx_label] = i
#     return new_labels, unique_labels



def epoch_step(loader, model, criterion):
    tqdm_str = ""
    batch_tqdm = tqdm.tqdm(total=len(loader), desc=f"Batch {tqdm_str}" , position=0, leave=True)
    # error_file = f'{save_path}/{tqdm_str}_error.txt' if save_path else None
    epoch_loss = []
    top1_epoch, top3_epoch, top5_epoch = [], [], []
    for batch_step, data in enumerate(loader):
        if batch_step == len(loader)-1:
            break   
        len_videos = 1188
        
        loss_avg = []
        top1_avg, top3_avg, top5_avg = [], [], []
        # writer_path = f'{save_path}/{tqdm_str}_EPOCH_{epoch}_video_{batch_step}.mp4' if save_path else None
        # if save_path and not os.path.exists(writer_path):
        #     os.makedirs(os.path.dirname(writer_path), exist_ok=True)


        # n_images = 5 if len(data) == 2 else 3
        # video_writer = cv2.VideoWriter(writer_path, fourcc, 30, (n_images*346,260)) if (not train or batch_step % 10==0 and save_path) else None
        loss = 0
        
        predictions = []
        GT = []
        labels = []
        for t in range(1,len_videos -1 ):
            events, depth, mask = get_data(data, t)
            kerneled = None
            
            with autocast(device_type="cuda"):
                encoding, pred = model(events, mask)
                training_step += 1
                predictions.append(pred)
                GT.append(encoding.detach().clone())
                labels.append(torch.arange(encoding.shape[0], device=device))
                predictions = torch.stack(predictions[0:-1], dim=0)
                ground_truths = torch.stack(GT[1:], dim=0)
                with torch.cuda.amp.autocast(enabled=False):
                    target, score = compute_CPC_loss(predictions, ground_truths, criterion)
                    score /= model.temperature
                    
                    target = target.argmax(dim=1)
                    if criterion:
                        loss = criterion(score, target)
                top1, top3, top5 = calc_topk_accuracy(score, target, (1, 3, 5))
                loss_avg.append(loss.item())
                top1_avg.append(top1.item())
                top3_avg.append(top3.item())
                top5_avg.append(top5.item())
                del score, target
                predictions = []
                
                training_step = 0
                if model.model_type != "Transformer":
                    model.detach_states()
                loss = 0

            del  encoding, depth, events, kerneled
            break
        ## shuffle old_gt
        
        
        epoch_loss.append(sum(loss_avg)/len(loss_avg))
        top1_epoch.append(sum(top1_avg)/len(top1_avg))
        top3_epoch.append(sum(top3_avg)/len(top3_avg))
        top5_epoch.append(sum(top5_avg)/len(top5_avg))
        string_value = f"Batch {batch_step} / {len(loader)}, Loss: {epoch_loss[-1]}, Top1: {top1_epoch[-1]}, Top3: {top3_epoch[-1]}, Top5: {top5_epoch[-1]}, LR: {optimizer.param_groups[0]['lr']}"
        print(string_value)
        # video_writer.release() if video_writer else None   
        if model.model_type != "Transformer":
            model.reset_states()            
        batch_tqdm.update(1)
        batch_tqdm.set_postfix({"loss": epoch_loss[-1], "top1": top1_epoch[-1], "top3": top3_epoch[-1], "top5": top5_epoch[-1]})
           
    # stats.dump_stats("profiling_results.prof")
        batch_tqdm.close()
    return GT, labels

                
def plotly_TSNE(tsne, labels, video_label, name):
    ## nice plotly with labels on hover
    fig = plotly.graph_objs.Figure()
    for i in range(10):
        idx = labels == i
        fig.add_trace(
            plotly.graph_objs.Scatter(
                x=tsne[idx, :][:, 0],
                y=tsne[idx, :][:, 1],
                mode="markers",
                name=video_label[i],
                text=[str(video_label[i])] * tsne[idx, :][:, 0].shape[0],
                hoverinfo="text",
            )
        )

    ## save the htmtl plot
    
    fig.update_layout(title=name)
    plotly.offline.plot(fig, filename=name + ".html")

def run_TSNE(data_loader, model):
    with torch.no_grad():
        seqs, labs = epoch_step(data_loader, model, None)
    all_seq = torch.cat(seqs)
    all_labels = torch.cat(labs)
    unique_label = torch.unique(all_labels)
    tsne = TSNE(n_components=2, verbose=0, perplexity=50, max_iter=2000).fit_transform(
        all_seq.cpu().numpy()
    )
    ## only take first 10 labels
    plotly_TSNE(tsne, all_labels, unique_label, "TSNE of DHP19EPC")


if __name__ == "__main__":
    # Training settings
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print("Using GPU")
    else:
        print("Using CPU")

    main()

