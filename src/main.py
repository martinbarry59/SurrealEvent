from models.EventSurreal import EventToDepthModel
from models.UnetSurreal import UNet
from models.MobileNetSurreal import UNetMobileNet
from utils.dataloader import EventDepthDataset, collate_event_batches
import torch
from config import data_path
import cv2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
def train(model, loader, optimizer, criterion):
    for batch in loader:
        events_videos, mask_videos, depths = batch
        
        t = 0
        for events, mask, depth in zip(events_videos, mask_videos, depths):
            loss = 0
            
            optimizer.zero_grad()
            batch_size = events.shape[0]
            current_depth = torch.zeros(batch_size, 1,260, 346, device=device)
            if t >= 10:
                # output = model(events, mask)
                output = model(events, current_depth)
                loss += criterion(output.squeeze(1), depth)
                
                # break
                current_depth = output
                y = events[0,:,1] * 346
                x = events[0,:,2] * 260
                if loss != 0:
                    loss.backward()
                optimizer.step()
                img = torch.zeros(260, 346, device=device)
                img[x.long(), y.long()] = 1
                merged = torch.cat([output[0,0],img, depth[0]], dim=1).cpu().detach().numpy()
                # merged = output.cpu().detach().numpy()
                cv2.imshow("depth", merged)
                cv2.waitKey(1)
            t += 1


def main(path):
    dataset = EventDepthDataset(data_path)
    loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_event_batches)
    width = 346
    height = 260
    # model = EventToDepthModel(embed_dim=128, hidden_dim=64, height=height, width=width)
    model = UNetMobileNet(in_channels = 3, out_channels = 1)
    model.to(device)

    criterion = torch.nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()

    for epoch in range(1000):
        train(model, loader, optimizer, criterion)
    

if __name__ == "__main__":
    
    
    path = "/home/martin-barry/Desktop/HES-SO/Event-Based/Surreal/dataset/05_05/05_05_c0002"
    
    main(path)
