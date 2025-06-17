# -*- coding: utf-8 -*-
# @Time    : 2022/6/10 14:18
# @Author  : Martin Barry

from __future__ import print_function
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
## add parent folder to path
from utils.nn_utils import init_simulation, sequence_for_LSTM
import torch
import cv2
import tqdm
import os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or use 'XVID' for .avi

def run_TSNE(model, loader, criterion = None):
    
    batch_tqdm = tqdm.tqdm(total=len(loader), desc=f"" , position=0, leave=True)
    epoch_loss = []
    top1_epoch, top3_epoch, top5_epoch = [], [], []
    for batch_step, data in enumerate(loader):
        # if batch_step == len(loader)-1:
        #     break   
        N = 100
        # loss_avg, top1_avg, top3_avg, top5_avg, TMP_OLD_GT = regular_steps(data, model, optimizer, criterion, train=train, scaler=scaler)
        loss_avg, top1_avg, top3_avg, top5_avg = sequence_for_LSTM(data, model, criterion, None, device, False, None, training_steps=N, block_update=N, rand_step= False)

        epoch_loss.append(sum(loss_avg)/len(loss_avg))
        top1_epoch.append(sum(top1_avg)/len(top1_avg))
        top3_epoch.append(sum(top3_avg)/len(top3_avg))
        top5_epoch.append(sum(top5_avg)/len(top5_avg))
        string_value = f"Batch {batch_step} / {len(loader)}, Loss: {epoch_loss[-1]}, Top1: {top1_epoch[-1]}, Top3: {top3_epoch[-1]}, Top5: {top5_epoch[-1]}"
        print(string_value)
        if "Trans" not in model.model_type:
            model.reset_states()            
        batch_tqdm.update(1)
        batch_tqdm.set_postfix({"loss": epoch_loss[-1], "top1": top1_epoch[-1], "top3": top3_epoch[-1], "top5": top5_epoch[-1]})
        
# stats.dump_stats("profiling_results.prof")
    batch_tqdm.close()
    return sum(epoch_loss)/len(epoch_loss)
def main():
    with torch.no_grad():
        model, train_loader, _, _, _, \
        criterion, _, _ = init_simulation(device, batch_size=10, network="Transformer", checkpoint_file = f"model_epoch_5_TransLSTM.pth")
        run_TSNE(model, train_loader, criterion)
    





if __name__ == "__main__":
    # Training settings
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print("Using GPU")
    else:
        print("Using CPU")

    main()

