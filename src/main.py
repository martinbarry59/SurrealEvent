
from models.BOBW import BestOfBothWorld
from utils.dataloader import EventDepthDataset, CNN_collate, Transformer_collate

import torch
from config import data_path, checkpoint_path, results_path
import cv2
import tqdm
import os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or use 'XVID' for .avi

import torch.nn.functional as F
import lpips
import torchvision.transforms as T
from ignite.metrics import SSIM
from ignite.engine import Engine
from utils.nn_utils import sequence_for_LSTM
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
def preprocess(batch):
    transforms = T.Compose(
        [
            T.Normalize(mean=0.5, std=0.5),  # map [0, 1] into [-1, 1]
            T.Resize(size=(264,352)),
        ]
    )
    batch = transforms(batch)
    return batch

def postprocess(batch):
    batch = (batch + 1) / 2
    batch = torch.nn.functional.interpolate(batch, size=(260, 346), mode='bilinear', align_corners=False)
    return batch
class Warper:
    def __init__(self, shapes):
        super(Warper, self).__init__()
        N, C, H, W = shapes
        grid_y, grid_x = torch.meshgrid(torch.arange(0, H), torch.arange(0, W), indexing='ij')
        self.grid = torch.stack((grid_x, grid_y), 2).float().cuda()  # [H, W, 2]
        self.grid = self.grid[None].repeat(N, 1, 1, 1)  # [N, H, W, 2]
    def warp_image(self,img, flow):
        
        flow = flow.permute(0, 2, 3, 1)  # [N, H, W, 2]
        new_grid = self.grid + flow
        # Normalize to [-1, 1]
        new_grid[..., 0] = 2.0 * new_grid[..., 0] / (W - 1) - 1.0
        new_grid[..., 1] = 2.0 * new_grid[..., 1] / (H - 1) - 1.0
        warped = F.grid_sample(img.float(), new_grid, mode='bilinear', padding_mode='border', align_corners=True)
        return warped

    # create default evaluator for doctests

    

    
def final_inference(model, loader, criterion, save_path=None):
    model.eval()
    def eval_step(engine, batch):
        return batch

    default_evaluator = Engine(eval_step)
    SSIM_metric = SSIM(data_range=1.0)
    SSIM_metric.attach(default_evaluator, 'ssim')
    with torch.no_grad():
        tqdm_str = "inference"
        batch_tqdm = tqdm.tqdm(total=len(loader), desc=f"Batch {tqdm_str}" , position=0, leave=True)
        error_file = f'{save_path}/{tqdm_str}_error.txt' if save_path else None
        inference_loss = []
        inference_loss_MSE = []
        inference_loss_SSIM = []
        
        for batch_step, data in enumerate(loader):
            len_videos = 1188
            writer_path = f'{save_path}/{tqdm_str}_video_{batch_step}.mp4' if save_path else None
            loss_avg = []
            loss_MSE = []
            loss_SSIM = []
           
            
            for t in range(1, len_videos):
                events, depth, mask = get_data(data, t)
                outputs, _ = model(events, mask)

                    ## repeat ouputs for 3 channels
                instant_loss = criterion(outputs.repeat(1, 3, 1, 1), depth.unsqueeze(1).repeat(1, 3, 1, 1)).mean()

                ## also compute MSE loss
                MSE_loss = F.mse_loss(outputs, depth.unsqueeze(1))
                ## compute SSIM loss
                state = default_evaluator.run([[outputs, depth.unsqueeze(1)]])
                loss_avg.append(instant_loss.item())
                loss_MSE.append(MSE_loss.item())
                loss_SSIM.append(state.metrics['ssim'])                    
                # if video_writer:
                #     images_to_write = [events, depth[0], outputs[0,0].squeeze(0)]
                #     add_frame_to_video(video_writer, images_to_write)
            losses = f"Batch Losses, Loss: {sum(loss_avg)/len(loss_avg)}, MSE: {sum(loss_MSE)/len(loss_MSE)}, SSIM: {sum(loss_SSIM)/len(loss_SSIM)}"
            batch_tqdm.update(1)
            batch_tqdm.set_postfix({"loss": instant_loss})
            inference_loss.append(sum(loss_avg)/len(loss_avg))
            inference_loss_MSE.append(sum(loss_MSE)/len(loss_MSE))
            inference_loss_SSIM.append(sum(loss_SSIM)/len(loss_SSIM))
            with open(error_file, "a") as f:
                f.write(f"{losses}\n")
            model.reset_states()
        final_losses = f"Final Losses, Loss: {sum(inference_loss)/len(inference_loss)}, MSE: {sum(inference_loss_MSE)/len(inference_loss_MSE)}, SSIM: {sum(inference_loss_SSIM)/len(inference_loss_SSIM)}"
        with open(error_file, "a") as f:
            f.write(f"{final_losses}\n")
def evaluation(model, loader, optimizer, epoch, criterion = None, train=True, save_path=None, scaler=None):

    model.train() if train else model.eval()


    with torch.set_grad_enabled(train):
        tqdm_str = train *"training" + (1-train) * "testing"
        batch_tqdm = tqdm.tqdm(total=len(loader), desc=f"Batch {tqdm_str}" , position=0, leave=True)
        error_file = f'{save_path}/{tqdm_str}_error.txt' if save_path else None
        epoch_loss = []
        loss_MSE = []
        loss_SSIM = []
        for batch_step, data in enumerate(loader):
            


            writer_path = f'{save_path}/{tqdm_str}_EPOCH_{epoch}_video_{batch_step}.mp4' if save_path else None
            if save_path and not os.path.exists(writer_path):
                os.makedirs(os.path.dirname(writer_path), exist_ok=True)
            video_writer = cv2.VideoWriter(writer_path, fourcc, 30, (3*346,260)) if (not train or batch_step % 10==0 and save_path) else None
            with torch.cuda.amp.autocast():
                loss_avg, loss_MSE, loss_SSIM= sequence_for_LSTM(data, model, criterion, optimizer, device, train, scaler,video_writer = video_writer)
            
                batch_loss = sum(loss_avg)/len(loss_avg)
                epoch_loss.append(batch_loss)
                if len(loss_MSE) > 0:
                    batch_loss_MSE = sum(loss_MSE)/len(loss_MSE)
                    loss_MSE.append(batch_loss_MSE)
                    batch_loss_SSIM = sum(loss_SSIM)/len(loss_SSIM)
                    loss_SSIM.append(batch_loss_SSIM)

                with open(error_file, "a") as f:
                    f.write(f"Epoch {epoch}, Batch {batch_step} / {len(loader)}, Loss: {batch_loss}, LR: {optimizer.param_groups[0]['lr']}\n")
                video_writer.release() if video_writer else None   
                if model.model_type != "Transformer":
                    model.reset_states()            
                batch_tqdm.update(1)
            batch_tqdm.set_postfix({"loss": batch_loss})
           
        batch_tqdm.close()
    if len(loss_MSE) > 0:
        print(f"Epoch {epoch}, Loss: {sum(epoch_loss)/len(epoch_loss)}, MSE: {sum(loss_MSE)/len(loss_MSE)}, SSIM: {sum(loss_SSIM)/len(loss_SSIM)}")
    return sum(epoch_loss)/len(epoch_loss)

def main():
    batch_size = 12
    network = "BOBWLSTM" # LSTM, Transformer, BOBWFF, BOBWLSTM
    

    loading_method = CNN_collate if (not ((network =="Transformer") or ("BOBW" in network))) else Transformer_collate
    train_dataset = EventDepthDataset(data_path+"/train/", tsne=False)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=loading_method)
    test_dataset = EventDepthDataset(data_path+"/test/", tsne=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=10 * batch_size, shuffle=False, collate_fn=loading_method)
    epoch_checkpoint = 0
    save_path = f"{results_path}/{network}"
    model = BestOfBothWorld(model_type=network)
    if checkpoint_path:
        checkpoint_file = f'{checkpoint_path}/model_epoch_4_{network}.pth'
        
        print(f"Loading checkpoint from {checkpoint_file}")
        try:
            model.load_state_dict(torch.load(checkpoint_file, map_location=device))
            epoch_checkpoint = int(checkpoint_file.split("_")[2]) + 1
        except:
            print("Checkpoint not found, starting from scratch")
    model.to(device)

    # criterion = torch.nn.SmoothL1Loss()
    criterion = lpips.LPIPS(net='alex')
    criterion.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)  # 10 = total number of epochs
    test_only = False
    min_loss = float('inf')
    for epoch in range(100):
        if epoch >= epoch_checkpoint:
            scaler = torch.amp.GradScaler(device=device)   
            if not test_only:     
 
                train_loss = evaluation(model, train_loader, optimizer, epoch, criterion, train=True, save_path=save_path, scaler=scaler)
                test_loss = evaluation(model, test_loader, optimizer, epoch, criterion= criterion, train=False, save_path=save_path , scaler=scaler)

                if test_loss < min_loss and not test_only:
                    min_loss = test_loss
                    torch.save(model.state_dict(), f'{checkpoint_path}/model_epoch_{epoch}_{model.model_type}_best.pth')
                torch.save(model.state_dict(), f'{checkpoint_path}/model_epoch_{epoch}_{model.model_type}.pth')

            ## divide by 10 the lr at each epocj

                print(f"Epoch {epoch + epoch_checkpoint}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
                exit()
            else:
                test_loss = final_inference(model, test_loader, criterion, save_path=save_path)
                break
        scheduler.step()
if __name__ == "__main__":
    
    main()
    