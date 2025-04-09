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

def seq_evaluation(model, loader, optimizer, epoch, criterion = None, train=True):
    model.train() if train else model.eval()
    with torch.set_grad_enabled(train):
        batch_step = 0
        tqdm_str = train *" Training:" + (1-train) * "Testing:"
        batch_tqdm = tqdm.tqdm(total=len(loader), desc=f"Batch {tqdm_str}:" , position=0, leave=True)
        for batch in loader:
            events_videos, depths = batch
            loss_avg = []
            video_writer = cv2.VideoWriter(f'{videos_path}/test_evaluation_video_{batch_step}_EPOCH_{epoch}.mp4', fourcc, 30, (3*depths.shape[3], depths.shape[2])) #if not train else None
            t = 0
            loss = 0
            for events, depth in zip(events_videos, depths):
                events = events.to(device)
                depth = depth.to(device)
                
                
                optimizer.zero_grad()

                if t >= 10:
                    outputs = model(events)
                    loss += criterion(outputs.squeeze(1), depth)
                    loss_avg.append(loss.detach().item())
                    if train and t% 20 == 0:
                        ## change dim 1 with dim 2
                        loss.backward()
                        optimizer.step()
                        model.convlstm.detach_hidden()
                        loss = 0

                    if video_writer:
                        add_frame_to_video(video_writer, events.cpu(), depth[0].cpu(), outputs[0,0].squeeze(0).cpu())
                    
                    del  outputs
                t += 1
            video_writer.release() if video_writer else None   
            batch_step += 1
            model.convlstm.reset_hidden()            
            batch_tqdm.update(1)
            batch_tqdm.set_postfix({"loss": sum(loss_avg)/len(loss_avg)})
    return sum(loss_avg)/len(loss_avg)
def evaluation(model, loader, optimizer, epoch, criterion = None, train=True):
    model.train() if train else model.eval()
    with torch.set_grad_enabled(train):
        batch_step = 0
        epoch_loss = 0
        for batch in loader:
            events_videos, depths = batch
            loss_avg = []
            t = 0
            frame_tqdm = tqdm.tqdm(total=len(events_videos), desc="Processing Videos", position=0, leave=True)
            video_writer = cv2.VideoWriter(f'{videos_path}/test_evaluation_video_{batch_step}_EPOCH_{epoch}.mp4', fourcc, 30, (3*depths.shape[3], depths.shape[2])) if not train else None
            loss = 0
            for events, depth in zip(events_videos, depths):
                    
                
                events = events.to(device)
                depth = depth.to(device)
                
                
                optimizer.zero_grad()
                batch_size = events.shape[0]
                current_depth = torch.zeros(batch_size, 1,260, 346, device=device)

                if t >= 10:
                    output = model(events, current_depth)
                    loss += criterion(output.squeeze(1), depth)
                    
                    # break
                    current_depth = output
                    y = events[0,:,1] * 346
                    x = events[0,:,2] * 260
                    loss_avg.append(loss.detach().item())
                    ## print t and flush
                    frame_tqdm.update(1)
                    frame_tqdm.set_postfix({"loss": loss.item()})
                    if train and t % 5 == 0:
                        if loss != 0:
                            loss.backward()
                        optimizer.step()
                        loss = 0
                    if video_writer:
                        add_frame_to_video(video_writer, events, depth[0], output[0,0])
                    del events, depth
                    
                t += 1
                if t > 200:
                    break
            frame_tqdm.close()
            epoch_loss += sum(loss_avg)/len(loss_avg)
            video_writer.release() if video_writer else None
            print("bideo writer released")
            
            batch_step += 1
            
            print(f"Batch Step: {batch_step} / {len(loader)}, Loss: {sum(loss_avg)/len(loss_avg):.4f}")
            del events_videos, mask_videos, depths
            break
    return epoch_loss

def main():
    batch_size = 20
    train_dataset = EventDepthDataset(data_path+"/train/")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_event_batches)
    test_dataset = EventDepthDataset(data_path+"/test/")
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_event_batches)
    # width = 346
    # height = 260
    model = UNetMobileNetLSTM(in_channels = 2, out_channels = 1)
    if checkpoint_path:
        try:
            model.load_state_dict(torch.load(checkpoint_path))
        except:
            print("Checkpoint not found, starting from scratch")
    model.to(device)

    criterion = torch.nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)
    
    min_loss = float('inf')
    for epoch in range(1000):
        train_loss = seq_evaluation(model, train_loader, optimizer, epoch, criterion)
        test_loss = seq_evaluation(model, test_loader, optimizer, epoch, criterion= criterion, train=False)
        if test_loss < min_loss:
            min_loss = test_loss
            torch.save(model.state_dict(), f'{checkpoint_path}/model_epoch_{epoch}.pth')
        print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
    

if __name__ == "__main__":
        
    main()
