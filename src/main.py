from models.MobileNetSurreal import UNetMobileNetSurreal
from utils.dataloader import EventDepthDataset, vectorized_collate
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
print(device)
def edge_aware_loss(pred, target):
    
    pred_dx = pred[ :, :, 1:] - pred[ :, :, :-1]
    target_dx = target[:, :, 1:] - target[:, :, :-1]
    pred_dy = pred[ :, 1:, :] - pred[ :, :-1, :]
    target_dy = target[ :, 1:, :] - target[ :, :-1, :]
    return F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)

def evaluation(model, loader, optimizer, epoch, criterion = None, train=True, save_path=None):
    model.train() if train else model.eval()
    scaler = torch.amp.GradScaler(device=device)
    with torch.set_grad_enabled(train):
        tqdm_str = train *" training" + (1-train) * "testing"
        batch_tqdm = tqdm.tqdm(total=len(loader), desc=f"Batch {tqdm_str}" , position=0, leave=True)
        error_file = f'{save_path}/{tqdm_str}_error.txt' if save_path else None
        for batch_step, batch in enumerate(loader):
            events_videos, depths = batch
            loss_avg = [0]
            writer_path = f'{save_path}/{tqdm_str}_EPOCH_{epoch}_video_{batch_step}.mp4' if save_path else None
            if save_path and not os.path.exists(writer_path):
                os.makedirs(os.path.dirname(writer_path), exist_ok=True)
            video_writer = cv2.VideoWriter(writer_path, fourcc, 30, (5*depths.shape[3], depths.shape[2])) if (not train or batch_step % 20==0 and save_path) else None
            loss = 0
            block_update = 60
            N_update = 1
            t_start = random.randint(10, 1188 - N_update * block_update)
            t_end = t_start + N_update * block_update
            previous_output = None
            for t, (events, depth) in enumerate(zip(events_videos, depths)):
                events = events.to(device)
                depth = depth.to(device)
                with autocast(device_type="cuda"):
                    if t < t_start:
                        with torch.no_grad():
                            outputs, _, kerneled = model(events)
                    else:
                        optimizer.zero_grad()
                        outputs, _, kerneled = model(events)

                instant_loss = criterion(outputs.squeeze(1), depth)
                loss_avg.append(instant_loss.item())
                if t >= t_start:
                    loss += instant_loss
                    loss += F.smooth_l1_loss(outputs, previous_output)
                    loss += edge_aware_loss(outputs.squeeze(1), depth)
                    
                    if train and t % block_update == 0:
                        ## change dim 1 with dim 2
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                        scaler.step(optimizer)
                        scaler.update()
                        model.detach_states()
                        loss = 0
                with torch.no_grad():
                    if video_writer:
                        kerneled = kerneled-kerneled.min() 
                        kerneled = kerneled/(kerneled.max()+1e-6)

                        images_to_write = [events, kerneled[0,0], kerneled[0,1], depth[0], outputs[0,0].squeeze(0)]
                        add_frame_to_video(video_writer, images_to_write)
                previous_output = outputs.detach().clone()
                del  outputs, depth, events, kerneled
                if t == t_end:
                    break
            with open(error_file, "a") as f:
                f.write(f"Epoch {epoch}, Batch {batch_step} / {len(loader)}, Loss: {sum(loss_avg)/len(loss_avg)}, LR: {optimizer.param_groups[0]['lr']}\n")
            video_writer.release() if video_writer else None   
            model.reset_states()            
            batch_tqdm.update(1)
            batch_tqdm.set_postfix({"loss": sum(loss_avg)/len(loss_avg)})
        batch_tqdm.close()
    return sum(loss_avg)/len(loss_avg)

def main():
    batch_size = 15


    train_dataset = EventDepthDataset(data_path+"/train/")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=vectorized_collate)
    test_dataset = EventDepthDataset(data_path+"/test/")
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=vectorized_collate)


    # Load the model UNetMobileNetSurreal
    use_lstm = False
    method = "concatenate" ## add or concatenate
    path_str = use_lstm *"LSTM" + (1-use_lstm) * "FF"
    save_path = f'{results_path}/{path_str}_{method}'
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path)
    model = UNetMobileNetSurreal(in_channels = 2 + 1 * (not use_lstm), out_channels = 1, use_lstm = use_lstm, method = method) ## if no LSTM use there we give previous output as input
    
    if checkpoint_path:
        checkpoint_file = f'{checkpoint_path}/model_epoch_0_{path_str}_{method}.pth'
        try:
            model.load_state_dict(torch.load(checkpoint_file))
        except:
            print("Checkpoint not found, starting from scratch")
    model.to(device)

    criterion = torch.nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)  # 10 = total number of epochs



    min_loss = float('inf')
    for epoch in range(10):
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
