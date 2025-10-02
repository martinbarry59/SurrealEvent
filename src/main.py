
from models.ConvLSTM import EConvlstm
from utils.dataloader import EventDepthDataset, Transformer_collate

import torch
from config import data_path, checkpoint_path, results_path
import cv2
import tqdm
import os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or use 'XVID' for .avi

import lpips

from utils.nn_utils import sequence_for_LSTM
import glob



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
            video_writer = cv2.VideoWriter(writer_path, fourcc, 30, (3*346,260)) if (not train or batch_step % 100==0 and save_path) else None
            # with torch.amp.autocast(device_type=device.type):
            loss_avg, loss_MSE, loss_SSIM, step_size, zero_run = sequence_for_LSTM(data, model, criterion, optimizer, device, train, epoch, scaler, video_writer=video_writer)

            batch_loss = sum(loss_avg)/len(loss_avg)
            epoch_loss.append(batch_loss)
            if len(loss_MSE) > 0:
                batch_loss_MSE = sum(loss_MSE)/len(loss_MSE)
                loss_MSE.append(batch_loss_MSE)
                batch_loss_SSIM = sum(loss_SSIM)/len(loss_SSIM)
                loss_SSIM.append(batch_loss_SSIM)

            with open(error_file, "a") as f:
                f.write(f"Epoch {epoch}, Batch {batch_step} / {len(loader)}, Loss: {batch_loss}, LR: {optimizer.param_groups[0]['lr']}, Step Size: {step_size}\n")
            video_writer.release() if video_writer else None
            if model.model_type != "Transformer":
                model.reset_states()            
            batch_tqdm.update(1)
            batch_tqdm.set_postfix({"loss": batch_loss, "step_size": step_size, "zero_run": zero_run})
           
        batch_tqdm.close()
    if len(loss_MSE) > 0:
        print(f"Epoch {epoch}, Loss: {sum(epoch_loss)/len(epoch_loss)}, MSE: {sum(loss_MSE)/len(loss_MSE)}, SSIM: {sum(loss_SSIM)/len(loss_SSIM)}")
    return sum(epoch_loss)/len(epoch_loss)

def main():

    batch_train = 15
    batch_test = 100
    network = "CONVLSTM" # LSTM, Transformer, BOBWFF, BOBWLSTM
    
    ## set seed for reproducibility
    # torch.manual_seed(42)
    # torch.cuda.manual_seed(42)
    train_dataset = EventDepthDataset(data_path+"/train/", tsne=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_train, shuffle=True, collate_fn=Transformer_collate)
    test_dataset = EventDepthDataset(data_path+"/test/", tsne=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size= batch_test, shuffle=False, collate_fn=Transformer_collate)
    epoch_checkpoint = 0
    save_path = f"{results_path}/{network}"
    checkpoint_file = None
    if checkpoint_path:
        checkpoint_files = glob.glob(f'{checkpoint_path}/model_epoch_*_{network}.pth')
        print(checkpoint_files)
    if checkpoint_files:
        # Extract epoch numbers and find the file with the highest epoch
        def extract_epoch(fp):
            try:
                print(fp, os.path.basename(fp).split("_")[2])
                return int(os.path.basename(fp).split("_")[2])
            except Exception:
                return -1
        checkpoint_file = max(checkpoint_files, key=extract_epoch)
        
        model = EConvlstm(model_type=network, width=346, height=260)
        print(f"Loading checkpoint from {checkpoint_file}")
        try:
            model.load_state_dict(torch.load(checkpoint_file, map_location=device))
            epoch_checkpoint = extract_epoch(checkpoint_file) + 1
            print(f"Resuming from epoch {epoch_checkpoint}")
        except Exception as e:
            print(f"Checkpoint not found or failed to load: {e}\nStarting from scratch")
    else:
        print("No checkpoint files found, starting from scratch")
        model = EConvlstm(model_type=network, width=346, height=260) if "CONVLSTM" in network else BestOfBothWorld(model_type=network, width=346, height=260, embed_dim=256, depth=12, heads=8, num_queries=64)
    model.to(device)

    # criterion = torch.nn.SmoothL1Loss()
    criterion = lpips.LPIPS(net='alex')
    criterion.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)  # 10 = total number of epochs
    test_only = False
    save_path = save_path+ f"/{checkpoint_file.split('/')[-1].split('.')[0]}" if test_only else save_path
    min_loss = float('inf')
    for epoch in range(100):
        if epoch >= epoch_checkpoint:
            scaler = torch.amp.GradScaler(device=device)   
            if not test_only:

                train_loss = evaluation(model, train_loader, optimizer, epoch, criterion, train=True, save_path=save_path, scaler=scaler)
                test_loss = evaluation(model, test_loader, optimizer, epoch, criterion= criterion, train=False, save_path=save_path , scaler=scaler)

                save_string = f'{checkpoint_path}/model_epoch_{epoch}_{model.model_type}'
                if checkpoint_file is not None and "small" in checkpoint_file:
                    save_string += "_small"
                if test_loss < min_loss and not test_only:
                    min_loss = test_loss

                    torch.save(model.state_dict(), f'{save_string}_best.pth')
                torch.save(model.state_dict(), f'{save_string}.pth')


                print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
                exit()
            else:
                test_loss =  evaluation(model, test_loader, optimizer, epoch, criterion= criterion, train=False, save_path=save_path , scaler=scaler)
                break
        scheduler.step()
if __name__ == "__main__":
    
    main()
    