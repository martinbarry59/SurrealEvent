from models.MobileNetSurreal import UNetMobileNetSurreal
from models.TransformerEventSurreal import EventTransformer
from models.BOBW import BestOfBothWorld
from utils.dataloader import EventDepthDataset, CNN_collate, Transformer_collate
from utils.functions import add_frame_to_video
import torch
from config import data_path, results_path, checkpoint_path
import cv2
import tqdm
import os
import random
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or use 'XVID' for .avi
from torch.amp import autocast
import torch.nn.functional as F
import shutil
import lpips

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

def evaluation(model, loader, optimizer, epoch, criterion = None, train=True, save_path=None):
    model.train() if train else model.eval()
    scaler = torch.amp.GradScaler(device=device)
    with torch.set_grad_enabled(train):
        tqdm_str = train *"training" + (1-train) * "testing"
        batch_tqdm = tqdm.tqdm(total=len(loader), desc=f"Batch {tqdm_str}" , position=0, leave=True)
        error_file = f'{save_path}/{tqdm_str}_error.txt' if save_path else None
        epoch_loss = []
        
        for batch_step, data in enumerate(loader):
            len_videos = 1188
            
            loss_avg = [0]
            writer_path = f'{save_path}/{tqdm_str}_EPOCH_{epoch}_video_{batch_step}.mp4' if save_path else None
            if save_path and not os.path.exists(writer_path):
                os.makedirs(os.path.dirname(writer_path), exist_ok=True)


            n_images = 5 if len(data) == 2 else 3
            video_writer = cv2.VideoWriter(writer_path, fourcc, 30, (n_images*346,260)) if (not train or batch_step % 10==0 and save_path) else None
            loss = 0
            block_update = 100

            N_update = int(5 / block_update)
            t_start = random.randint(10, len_videos - N_update * block_update)
            t_end = t_start + N_update * block_update
            previous_output = None

            for t in range(len_videos):
                events, depth, mask = get_data(data, t)
                kerneled = None
                
                with autocast(device_type="cuda"):
                    if t < t_start:
                        with torch.no_grad():
                            if mask is not None:
                                outputs, attn_weights = model(events, mask)
                            else:
                                outputs, _, kerneled = model(events)
                    else:
                        
                        optimizer.zero_grad()
                        if mask is not None:
                            outputs, attn_weights = model(events, mask)
                        else:
                            outputs, _, kerneled = model(events)
                ## repeat ouputs for 3 channels
                instant_loss = criterion(outputs.repeat(1, 3, 1, 1), depth.unsqueeze(1).repeat(1, 3, 1, 1)).sum()
                loss_avg.append(instant_loss.item())
                if t >= t_start:
                    loss += instant_loss
                    # loss += F.smooth_l1_loss(outputs, previous_output)
                    # loss += edge_aware_loss(outputs.squeeze(1), depth)
                    if train and (((t-t_start) % block_update == (block_update-1)) or t == t_end-1):
                        ## change dim 1 with dim 2
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                        scaler.step(optimizer)
                        scaler.update()
                        if model.model_type != "Transformer":
                            model.detach_states()
                        loss = 0
                with torch.no_grad():
                    
                    if video_writer:
                        
                        if kerneled is not None:
                            kerneled = kerneled-kerneled.min() 
                            kerneled = kerneled/(kerneled.max()+1e-6)
                            images_to_write = [events, kerneled[0,0], kerneled[0,1], depth[0], outputs[0,0].squeeze(0)]
                        else:
                            images_to_write = [events, depth[0], outputs[0,0].squeeze(0)]
                        add_frame_to_video(video_writer, images_to_write)
                previous_output = outputs.detach().clone()
                
                if t == t_end -1:
                    # with torch.no_grad():
                    #     if video_writer:
                    #         for q in range(attn_weights.shape[1]):
                    #             plot_attention_map(attn_weights, events, 0, q, depth[0])
                    break
                del  outputs, depth, events, kerneled

            batch_loss = sum(loss_avg)/len(loss_avg)
            epoch_loss.append(batch_loss)
            with open(error_file, "a") as f:
                f.write(f"Epoch {epoch}, Batch {batch_step} / {len(loader)}, Loss: {batch_loss}, LR: {optimizer.param_groups[0]['lr']}\n")
            video_writer.release() if video_writer else None   
            if model.model_type != "Transformer":
                model.reset_states()            
            batch_tqdm.update(1)
            batch_tqdm.set_postfix({"loss": batch_loss})
        batch_tqdm.close()
    return sum(epoch_loss)/len(epoch_loss)

def main():
    batch_size = 20
    network = "BOBWLSTM" # LSTM, Transformer, BOBWFF, BOBWLSTM
    method = "add" ## add or concatenate


    loading_method = CNN_collate if (not ((network =="Transformer") or ("BOBW" in network))) else Transformer_collate
    train_dataset = EventDepthDataset(data_path+"/train/")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=loading_method)
    test_dataset = EventDepthDataset(data_path+"/test/")
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=loading_method)


    # Load the model UNetMobileNetSurreal
    
    save_path = f'{results_path}/{network}_{method}'
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path)
    if network == "Transformer":
        model = EventTransformer()
    elif "BOBW" in network:
        model = BestOfBothWorld(model_type=network)
    else:
        model = UNetMobileNetSurreal(in_channels = 2, out_channels = 1, net_type = network , method = method) ## if no LSTM use there we give previous output as input
    if checkpoint_path:
        checkpoint_file = f'{checkpoint_path}/model_epoch_1_{model.model_type}_{model.method}.pth'
        try:
            model.load_state_dict(torch.load(checkpoint_file))
        except:
            print("Checkpoint not found, starting from scratch")
    model.to(device)

    # criterion = torch.nn.SmoothL1Loss()
    criterion = lpips.LPIPS(net='alex')
    criterion.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)  # 10 = total number of epochs

    min_loss = float('inf')
    for epoch in range(100):
        train_loss = evaluation(model, train_loader, optimizer, epoch, criterion, train=True, save_path=save_path)
        test_loss = evaluation(model, test_loader, optimizer, epoch, criterion= criterion, train=False, save_path=save_path)
        scheduler.step()
        if test_loss < min_loss:
            min_loss = test_loss
            torch.save(model.state_dict(), f'{checkpoint_path}/model_epoch_{epoch}_{model.model_type}_{model.method}.pth')
        ## divide by 10 the lr at each epocj

        print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
    

if __name__ == "__main__":
        
    main()
