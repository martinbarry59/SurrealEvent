from models.MobileNetSurrealLSTM import UNetMobileNetLSTM
from models.MobileNetSurreal import UNetMobileNet
from utils.dataloader import EventDepthDataset, collate_event_batches
import torch
from config import data_path, videos_path, checkpoint_path
import cv2
import tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or use 'XVID' for .avi
print(device)
def add_frame_to_video(video_writer, events, depth, output):
    y = events[0,:,1] * 346
    x = events[0,:,2] * 260
    img = torch.zeros(260, 346)
    img[x.long(), y.long()] = 1
    merged = torch.cat([output,img, depth], dim=1).detach().numpy()
    merged = (merged * 255 ).astype('uint8')
    merged = cv2.cvtColor(merged, cv2.COLOR_GRAY2BGR)  # make it (H, W, 3)
    video_writer.write(merged)  # Write the frame to video

def evaluation(model, loader, optimizer, epoch, criterion = None, train=True, model_type = None):
    model.train() if train else model.eval()
    with torch.set_grad_enabled(train):
        batch_step = 0
        tqdm_str = train *" training" + (1-train) * "testing"
        batch_tqdm = tqdm.tqdm(total=len(loader), desc=f"Batch {tqdm_str}" , position=0, leave=True)
        for batch in loader:
            events_videos, depths = batch
            loss_avg = []
            video_writer = cv2.VideoWriter(f'{videos_path}/{tqdm_str}_evaluation_video_{batch_step}_EPOCH_{epoch}.mp4', fourcc, 30, (3*depths.shape[3], depths.shape[2])) if (not train or batch_step %50==0 )else None
            t = 0
            loss = 0
            batch_size = depths.shape[1]
            if not model_type == "LSTM":
                current_depth = torch.zeros(batch_size, 1,260, 346, device=device)
            for events, depth in zip(events_videos, depths):
                events = events.to(device)
                depth = depth.to(device)
                
                
                optimizer.zero_grad()

                if t >= 10:
                    if model_type == "LSTM":
                        outputs = model(events)
                    else:
                        outputs = model(events, current_depth)
                        current_depth = outputs
                    loss += criterion(outputs.squeeze(1), depth)
                    loss_avg.append(loss.detach().item())
                    if train and t% 5 == 0:
                        ## change dim 1 with dim 2
                        loss.backward()
                        optimizer.step()
                        if model_type == "LSTM":
                            model.convlstm.detach_hidden()
                        else:
                            current_depth = current_depth.detach()
                        loss = 0

                    if video_writer:
                        add_frame_to_video(video_writer, events.cpu(), depth[0].cpu(), outputs[0,0].squeeze(0).cpu())
                    
                    del  outputs
                t += 1
            video_writer.release() if video_writer else None   
            batch_step += 1
            if model_type == "LSTM":
                model.convlstm.reset_hidden()            
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
    model_type="FF" #"FF"
    if model_type == "LSTM":
        model = UNetMobileNetLSTM(in_channels = 2, out_channels = 1)
    elif model_type == "FF":
        model = UNetMobileNet(in_channels = 3, out_channels = 1)
    else:
        raise ValueError("Unknown model type")
    
    if checkpoint_path:
        try:
            model.load_state_dict(torch.load(checkpoint_path))
        except:
            print("Checkpoint not found, starting from scratch")
    model.to(device)

    criterion = torch.nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)

    min_loss = float('inf')
    for epoch in range(10):
        train_loss = evaluation(model, train_loader, optimizer, epoch, criterion, model_type=model_type)
        test_loss = evaluation(model, test_loader, optimizer, epoch, criterion= criterion, train=False, model_type=model_type)
        if test_loss < min_loss:
            min_loss = test_loss
            torch.save(model.state_dict(), f'{checkpoint_path}/model_epoch_{epoch}.pth')
        print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
    

if __name__ == "__main__":
        
    main()
