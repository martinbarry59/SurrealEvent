from models.MobileNetSurreal import UNetMobileNetSurreal
from utils.dataloader import EventDepthDataset, collate_event_batches
from utils.functions import add_frame_to_video
import torch
from config import data_path, videos_path, checkpoint_path
import cv2
import tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or use 'XVID' for .avi
print(device)


def evaluation(model, loader, optimizer, epoch, criterion = None, train=True):
    model.train() if train else model.eval()
    with torch.set_grad_enabled(train):
        batch_step = 0
        tqdm_str = train *" training" + (1-train) * "testing"
        batch_tqdm = tqdm.tqdm(total=len(loader), desc=f"Batch {tqdm_str}" , position=0, leave=True)
        for batch in loader:
            events_videos, depths = batch
            loss_avg = []
            video_writer = cv2.VideoWriter(f'{videos_path}/{tqdm_str}_EPOCH_{epoch}_video_{batch_step}_{model.model_type}_{model.method}.mp4', fourcc, 30, (3*depths.shape[3], depths.shape[2])) if (not train or batch_step %1==0 )else None
            t = 0
            loss = 0
            
            for events, depth in zip(events_videos, depths):
                events = events.to(device)
                depth = depth.to(device)
                
                
                optimizer.zero_grad()

                if t >= 10:
                    outputs, _ = model(events)


                    loss += criterion(outputs.squeeze(1), depth)
                    loss_avg.append(loss.detach().item())
                    if train and t% 5 == 0:
                        ## change dim 1 with dim 2
                        loss.backward()
                        optimizer.step()
                        model.detach_states()
                        loss = 0

                    if video_writer:
                        add_frame_to_video(video_writer, events.cpu(), depth[0].cpu(), outputs[0,0].squeeze(0).cpu())
                    
                    del  outputs
                t += 1
            video_writer.release() if video_writer else None   
            batch_step += 1
            model.reset_states()            
            batch_tqdm.update(1)
            batch_tqdm.set_postfix({"loss": sum(loss_avg)/len(loss_avg)})
        batch_tqdm.close()
    return sum(loss_avg)/len(loss_avg)

def main():
    batch_size = 2
    train_dataset = EventDepthDataset(data_path+"/train/")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_event_batches)
    test_dataset = EventDepthDataset(data_path+"/test/")
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_event_batches)

    # Load the model UNetMobileNetSurreal
    use_lstm = False
    method = "add"  
    model = UNetMobileNetSurreal(in_channels = 2 + 1 * (not use_lstm), out_channels = 1, use_lstm = use_lstm, method = method) ## if no LSTM use there we give previous output as input
    
    if checkpoint_path:
        try:
            model.load_state_dict(torch.load(checkpoint_path))
        except:
            print("Checkpoint not found, starting from scratch")
    model.to(device)

    criterion = torch.nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    min_loss = float('inf')
    for epoch in range(10):
        train_loss = evaluation(model, train_loader, optimizer, epoch, criterion)
        test_loss = evaluation(model, test_loader, optimizer, epoch, criterion= criterion, train=False)
        if test_loss < min_loss:
            min_loss = test_loss
            torch.save(model.state_dict(), f'{checkpoint_path}/model_epoch_{epoch}_{model.model_type}_{model.method}.pth')
        print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
    

if __name__ == "__main__":
        
    main()
