
from utils.nn_utils import sequence_for_LSTM
import torch
from config import  checkpoint_path
import cv2
import tqdm
from utils.nn_utils import init_simulation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or use 'XVID' for .avi


def evaluation(model, loader, optimizer, epoch, criterion = None, train=True, save_path=None):
    model.train() if train else model.eval()
    scaler = torch.amp.GradScaler(device=device)
    
    with torch.set_grad_enabled(train):
        tqdm_str = train *"training" + (1-train) * "testing"
        batch_tqdm = tqdm.tqdm(total=len(loader), desc=f"Batch {tqdm_str}" , position=0, leave=True)
        error_file = f'{save_path}/{tqdm_str}_error.txt' if save_path else None
        epoch_loss = []
        top1_epoch, top3_epoch, top5_epoch = [], [], []
        for batch_step, data in enumerate(loader):
            # if batch_step == len(loader)-1:
            #     break   
            
            # loss_avg, top1_avg, top3_avg, top5_avg, TMP_OLD_GT = regular_steps(data, model, optimizer, criterion, train=train, scaler=scaler)
            loss_avg, top1_avg, top3_avg, top5_avg = sequence_for_LSTM(data, model, criterion, optimizer, device, train, scaler)

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
    network = "BOBW"  # or "BOBW", "LSTM"
    model, train_loader, test_loader, optimizer, scheduler, \
    criterion, save_path, epoch_checkpoint = init_simulation(device, batch_size=2, network=network, checkpoint_file=f"model_epoch_110.pth")
    min_loss = float('inf')
    for epoch in range(100):
        if epoch >= epoch_checkpoint:
            
            train_loss = evaluation(model, train_loader, optimizer, epoch, criterion, train=True, save_path=save_path)
            test_loss = evaluation(model, test_loader, optimizer, epoch, criterion= criterion, train=False, save_path=save_path)
            if test_loss < min_loss:
                min_loss = test_loss
                torch.save(model.state_dict(), f'{checkpoint_path}/model_epoch_{epoch}_{model.model_type}_best.pth')
            torch.save(model.state_dict(), f'{checkpoint_path}/model_epoch_{epoch}_{model.model_type}.pth')

            ## divide by 10 the lr at each epocj

            print(f"Epoch {epoch + epoch_checkpoint}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
        scheduler.step()

if __name__ == "__main__":
    
    main()
    