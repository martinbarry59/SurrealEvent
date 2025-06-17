from models.TransformerEventSurreal import EventTransformer
from models.BOBW_CPC import BestOfBothWorld
from utils.dataloader import EventDepthDataset, CNN_collate, Transformer_collate
from utils.functions import calc_topk_accuracy
import torch
from config import data_path, results_path, checkpoint_path
import cv2
import tqdm
import os
import random
import plotly
from sklearn.manifold import TSNE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or use 'XVID' for .avi
from torch.amp import autocast
import torch.nn.functional as F
import shutil
import torchvision.transforms as T
def variance_loss(x, eps=1e-4):
    # x: [B, D]
    std = torch.sqrt(x.var(dim=0) + eps)
    return torch.mean(F.relu(1.0 - std))
def plot_attention_map(attn_weights, events, b, q, img):
    query_attention = attn_weights[b, q]  # shape: [N]
    coords = events[b, :, 1:3].cpu().numpy()  # normalized x, y
    xs = coords[:, 0] * img.shape[1]
    ys = coords[:, 1] * img.shape[0]
    plt.imshow(img.cpu().numpy(), cmap='gray')
    plt.axis('off')
    plt.scatter(xs, ys, c=query_attention.cpu().numpy(), cmap='viridis', s=5)
    plt.colorbar(label='Attention Weight')
    plt.title(f"Query {q} attention over events")
    plt.show()
def plotly_TSNE(tsne, labels, video_label, name):
    ## nice plotly with labels on hover
    fig = plotly.graph_objs.Figure()

    for i in range(min(10, len(video_label))):
        idx = labels == i
        fig.add_trace(
            plotly.graph_objs.Scatter(
                x=tsne[idx, :][:, 0],
                y=tsne[idx, :][:, 1],
                mode="markers",
                name=str(video_label[i].item()),
                text=[str(video_label[i].item())] * tsne[idx, :][:, 0].shape[0],
                hoverinfo="text",
            )
        )

    ## save the htmtl plot
    
    fig.update_layout(title=name)
    plotly.offline.plot(fig, filename=name + ".html")
def edge_aware_loss(pred, target):
    
    pred_dx = pred[ :, :, 1:] - pred[ :, :, :-1]
    target_dx = target[:, :, 1:] - target[:, :, :-1]
    pred_dy = pred[ :, 1:, :] - pred[ :, :-1, :]
    target_dy = target[ :, 1:, :] - target[ :, :-1, :]
    return F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)

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
def check_CPC_representation_collapse(predictions, GT, threshold=0.1):
    ##  check standard deviation of the predictions and GT
    print("Checking CPC representation collapse")
    std = torch.std(predictions, dim=1)
    print(f"Standard deviation of predictions: {std.mean().item()}")
    score = get_score(predictions, GT)
    print(f"Score: {score.mean().item()}, Std: {score.std().item()}")
def preprocess(batch):
    transforms = T.Compose(
        [
            T.Normalize(mean=0.5, std=0.5),  # map [0, 1] into [-1, 1]
            T.Resize(size=(264,352)),
        ]
    )
    batch = transforms(batch)
    return batch
def process_output(mask):
    """task mask as input, compute the target for contrastive loss"""
    (T1, B1, T2, B2) = mask.size()  # [B, P, SQ, B, N, SQ]
    target = mask == 1
    target = target * 1
    target.requires_grad = False
    return target, (B1, T1, B2, T2)
def postprocess(batch):
    batch = (batch + 1) / 2
    batch = torch.nn.functional.interpolate(batch, size=(260, 346), mode='bilinear', align_corners=False)
    return batch
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
def regular_steps(data, model, optimizer, criterion, train=True, scaler = None):
    len_videos = 1188
    loss = 0
    training_steps = 300
    block_update = 50
    step_size =  torch.randint(1, 10, (1,)).item()
    N_update = int(training_steps / block_update)
    t_start = random.randint(10, len_videos - N_update * block_update)
    t_end = t_start + N_update * block_update
    training_step = 0
    
    predictions = []
    GT = []
    TMP_OLD_GT = []
    loss_avg = []
    top1_avg, top3_avg, top5_avg = [], [], []
    for t in range(1,len_videos - step_size, step_size):
        events, depth, mask = get_data(data, t)
        kerneled = None
        
        with autocast(device_type=device.type):
            if t < t_start:
                with torch.no_grad():
                    encoding, pred = model(events, mask)
            else:
                optimizer.zero_grad()
                encoding, pred = model(events, mask)

            if t >= t_start:
                training_step += 1
                predictions.append(pred)
                GT.append(encoding.detach().clone())
                if training_step % block_update == 0:
                    predictions = torch.stack(predictions[0:-1], dim=0)
                    ground_truths = torch.stack(GT[1:], dim=0)
                    # with torch.cuda.amp.autocast(enabled=False):
                    target, score = compute_CPC_loss(predictions, ground_truths, criterion)
                    score /= 0.07
                    # if len(OLD_GT) > 0:
                    #     old_ground_truths = torch.stack(OLD_GT, dim=0)  
                    #     target_old, score_old= compute_CPC_loss(predictions, old_ground_truths, criterion, False)
                        
                    #     target = torch.cat((target, target_old), dim=1)
                        
                    #     score = torch.cat((score, score_old), dim=1)
                    target = target.argmax(dim=1)
                    var_loss = variance_loss(predictions)
                    loss = criterion(score, target) + var_loss * 0.1
                    top1, top3, top5 = calc_topk_accuracy(score, target, (1, 3, 5))
                    loss_avg.append(loss.item())
                    top1_avg.append(top1.item())
                    top3_avg.append(top3.item())
                    top5_avg.append(top5.item())
                    del score, target
                    if train:

                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    predictions = []
                    TMP_OLD_GT.extend(GT)
                    GT = []
                    training_step = 0
                    
                    loss = 0
                    
        if t >= t_end -1:
            # with torch.no_grad():
            #     if video_writer:
            #         for q in range(attn_weights.shape[1]):
            #             plot_attention_map(attn_weights, events, 0, q, depth[0])
            break
        del  encoding, depth, events, kerneled
        return loss_avg, top1_avg, top3_avg, top5_avg, TMP_OLD_GT

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
def sequence_for_LSTM(data, model, criterion, optimizer, train = True, scaler = None):
    len_videos = 1188
    training_steps = 300
    block_update = 30
    step_size =  torch.randint(1, 10, (1,)).item()
    N_update = int(training_steps / block_update)
    t_start = random.randint(10, len_videos - N_update * block_update)
    t_end = t_start + N_update * block_update
    
    loss_avg = []
    top1_avg, top3_avg, top5_avg = [], [], []
    for n in range(N_update):
        seq_events = []
        seq_masks = []
        start_seq = t_start + n * block_update * step_size
        if start_seq >= t_end - block_update * step_size -1:
            break
        for t in range(start_seq, start_seq + block_update * step_size, step_size):
            events, _, mask = get_data(data, t)
            seq_events.append(events)
            seq_masks.append(mask)
        
        predictions, encodings = model(seq_events, seq_masks)
        if n == 0:
            with torch.no_grad():
                labels = torch.arange(0, encodings.shape[0], device=device)
                ## repeat to all other dimensions
                labels = labels.unsqueeze(1).repeat(1, encodings.shape[1]).view(-1)
                unique_labels = torch.unique(labels)
                ## flatten encodings
                flattened_encodings = encodings.reshape(-1, encodings.shape[-1])
                # print(encodings.shape, predictions.shape, flattened_encodings.shape, labels.shape, unique_labels.shape)
                tsne = TSNE(n_components=2, verbose=0, perplexity=50, max_iter=2000).fit_transform(
                    flattened_encodings.cpu().numpy()
                )
                plotly_TSNE(tsne, labels.cpu().detach().numpy(), unique_labels, f"{model.model_type}")
        predictions = predictions.permute(1,0,2).contiguous()[:-1]
        encodings = encodings.permute(1,0,2).contiguous()[1:]
        
        target, score = compute_CPC_loss(predictions, encodings, criterion)
        target = target.argmax(dim=1)
        score /= 0.07
        # loss = criterion(score, target)

        loss = vicreg_loss(predictions, encodings)
        top1, top3, top5 = calc_topk_accuracy(score, target, (1, 3, 5))
        loss_avg.append(loss.item())
        top1_avg.append(top1.item())
        top3_avg.append(top3.item())
        top5_avg.append(top5.item())
        del score, target
        if train:

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        # if n == 0:
        #     check_CPC_representation_collapse(predictions, encodings)
    return loss_avg, top1_avg, top3_avg, top5_avg
def evaluation(model, loader, optimizer, epoch, criterion = None, train=True, save_path=None):
    model.train() if train else model.eval()
    scaler = torch.amp.GradScaler(device=device)

    with torch.set_grad_enabled(train):
        tqdm_str = train *"training" + (1-train) * "testing"
        batch_tqdm = tqdm.tqdm(total=len(loader), desc=f"Batch {tqdm_str}" , position=0, leave=True)
        error_file = f'{save_path}/{tqdm_str}_error.txt' if save_path else None
        epoch_loss = []
        top1_epoch, top3_epoch, top5_epoch = [], [], []
        OLD_GT = []
        for batch_step, data in enumerate(loader):
            # if batch_step == len(loader)-1:
            #     break   
            
            # loss_avg, top1_avg, top3_avg, top5_avg, TMP_OLD_GT = regular_steps(data, model, optimizer, criterion, train=train, scaler=scaler)
            loss_avg, top1_avg, top3_avg, top5_avg = sequence_for_LSTM(data, model, criterion, optimizer, train, scaler)

            epoch_loss.append(sum(loss_avg)/len(loss_avg))
            top1_epoch.append(sum(top1_avg)/len(top1_avg))
            top3_epoch.append(sum(top3_avg)/len(top3_avg))
            top5_epoch.append(sum(top5_avg)/len(top5_avg))
            string_value = f"Epoch {epoch}, Batch {batch_step} / {len(loader)}, Loss: {epoch_loss[-1]}, Top1: {top1_epoch[-1]}, Top3: {top3_epoch[-1]}, Top5: {top5_epoch[-1]}, LR: {optimizer.param_groups[0]['lr']}"
            with open(error_file, "a") as f:
                f.write(string_value+"\n")
            # video_writer.release() if video_writer else None  
            if "Trans" not in model.model_type:
                model.reset_states()            
            batch_tqdm.update(1)
            batch_tqdm.set_postfix({"loss": epoch_loss[-1], "top1": top1_epoch[-1], "top3": top3_epoch[-1], "top5": top5_epoch[-1]})
           
    # stats.dump_stats("profiling_results.prof")
        batch_tqdm.close()
    return sum(epoch_loss)/len(epoch_loss)

def main():
    batch_size = 20
    network = "Transformer" # LSTM, Transformer, BOBWFF, BOBWLSTM
    

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

        checkpoint_file = f'{checkpoint_path}/model_epoch_5_{model.model_type}.pth'
        
        print(f"Loading checkpoint from {checkpoint_file}")
        try:
            model.load_state_dict(torch.load(checkpoint_file))
            epoch_checkpoint = int(checkpoint_file.split("_")[2]) + 1
        except:
            print("Checkpoint not found, starting from scratch")
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    criterion.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-5)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)  # 10 = total number of epochs

    min_loss = float('inf')
    for epoch in range(100):
        if epoch >= epoch_checkpoint:
            
            train_loss = evaluation(model, train_loader, optimizer, epoch, criterion, train=True, save_path=save_path)
            test_loss = evaluation(model, test_loader, optimizer, epoch, criterion= criterion, train=False, save_path=save_path)
            best = "False"
            if test_loss < min_loss:
                min_loss = test_loss
                torch.save(model.state_dict(), f'{checkpoint_path}/model_epoch_{epoch}_{model.model_type}_best.pth')
            torch.save(model.state_dict(), f'{checkpoint_path}/model_epoch_{epoch}_{model.model_type}.pth')

            ## divide by 10 the lr at each epocj

            print(f"Epoch {epoch + epoch_checkpoint}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
        scheduler.step()

if __name__ == "__main__":
    
    main()
    