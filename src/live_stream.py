from models.BOBW import BestOfBothWorld
from config import checkpoint_path
import sys
# sys.path.append("/usr/lib/python3/dist-packages")
#from metavision_sdk_stream import Camera, CameraStreamSlicer, SliceCondition
import numpy as np
import cv2
import torch
from datetime import timedelta
#from metavision_sdk_cv import ActivityNoiseFilterAlgorithm
#from metavision_sdk_base import EventCDBuffer
import dv_processing as dv
class datviewer:
    def __init__(self, reader, model , width, height):
        self.reader = reader
        self.width = width
        self.height = height
        self.events = None

        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.window_name = "Event Frame"
        self.window = cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        self.slicer = dv.EventStreamSlicer()
        resolution = (width, height)
        self.filter = dv.noise.BackgroundActivityNoiseFilter(resolution, 
                                                        backgroundActivityDuration=timedelta(milliseconds=1))
        
        self.slicer.doEveryTimeInterval(timedelta(milliseconds=60),self.retrieveEvents)
    def retrieveEvents(self, events):
        self.events = events
    def generateEvents(self):
        return self.events
    def run(self):

        while self.reader.isRunning():
            self.events = None

    # Read batch of events
            events = self.reader.getNextEventBatch()
                
            if events is None or len(events)== 0:
                continue
            self.slicer.accept(events)
            if self.events is None or len(self.events) == 0:
                continue
            # events = slicer.generateEvents()
            self.filter.accept(self.events)
            filtered_events = self.filter.generateEvents()

            # In your event processing loop:
            
            # Create a black image
            img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            N = 5000
            # Draw events: polarity 1 as white, 0 as blue (or any color you like)
            filtered_events = filtered_events.numpy()
            if len(filtered_events)  == 0:
                continue
            xs = self.width - filtered_events["x"][:N] -1 
            ys = filtered_events["y"][:N]
            ps = filtered_events["polarity"][:N]* 2 - 1
            ts = filtered_events["timestamp"][:N]
            events_tensor = torch.stack(( torch.tensor(ts).float(), torch.tensor(xs / self.width).float(),
                                            torch.tensor(ys / self.height).float(), 
                                            torch.tensor(ps).float()), dim=1)  # [N, 4].long()
            ## from float 32 to float 16
            mask = torch.ones((events_tensor.shape[0],), dtype=torch.bool)
            seq_events = events_tensor.unsqueeze(0).unsqueeze(0).to(device)
            seq_masks = mask.unsqueeze(0).unsqueeze(0).to(device)
            predictions, encodings = self.model(seq_events, seq_masks)
            
            img[ys[ps == 1], xs[ps == 1]] = (255, 255, 255)  # White for polarity 1
            img[ys[ps == 0], xs[ps == 0]] = (255, 0, 0)      # Blue for polarity 0

            ## crop image to 246, 346
            # merge the predictions into the image
            

            if predictions is not None:
                pred = predictions[0, 0].detach().cpu().numpy()
                pred = (pred * 255).astype(np.uint8)
                pred = cv2.applyColorMap(pred, cv2.COLORMAP_JET)
                img = cv2.addWeighted(img, 0.5, pred, 0.5, 0)
                
                # resize twice the size
                img = cv2.resize(img, (640, 640), interpolation=cv2.INTER_LINEAR)
            cv2.imshow("Event Frame", img)
            key = cv2.waitKey(1)
            if key == 27:  # ESC to quit
                break
def print_time_interval(events: dv.EventStore, model,
                         filter, width, height, device):

    

    return events
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Metavision SDK Get Started sample.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-i', '--input-event-file',
        help="Path to input event file (RAW or HDF5). If not specified, the camera live stream is used.")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    network = "BOBWLSTM" 
    reader = dv.io.CameraCapture()
    width, height = reader.getEventResolution()
    resolution = (width, height)
    if checkpoint_path:
        checkpoint_file = f'{checkpoint_path}/model_epoch_7_BOBWLSTM.pth'
        if "small" in checkpoint_file:
            model = BestOfBothWorld(model_type=network, width=width, height = height,embed_dim=128, depth=6, heads=8, num_queries=16)
        else:
            model = BestOfBothWorld(model_type=network, width=width, height = height,embed_dim=256, depth=12, heads=8, num_queries=64)
        print(f"Loading checkpoint from {checkpoint_file}")
        try:
            model.load_state_dict(torch.load(checkpoint_file, map_location=device))
            epoch_checkpoint = int(checkpoint_file.split("_")[2]) + 1
        except:
            print("Checkpoint not found, starting from scratch")
    model.to(device)
    viewer = datviewer(reader, model, width, height)
    viewer.run()
    # if args.input_event_file:
    #     camera = Camera.from_file(args.input_event_file)
    # else:
    #     camera = Camera.from_first_available()
    # width, height = camera.width(), camera.height()
    # slice_condition = SliceCondition.make_n_us(100000)
    # slicer = CameraStreamSlicer(camera.move(), slice_condition=slice_condition)
    # activity_filter = ActivityNoiseFilterAlgorithm(width, height, 20000)
    

    # for slice in slicer:
        # events = slice.events
        # events_buf = EventCDBuffer()
        # activity_filter.process_events(slice.events, events_buf)
        # filtered_events = events_buf.numpy().copy()
        # xs = filtered_events[:N]['x']
        # ys = filtered_events[:N]['y']
        # ps = filtered_events[:N]['p']* 2 - 1
        # ts = filtered_events[:N]['t']
    
        

        

    cv2.destroyAllWindows()

if __name__ == "__main__":
    with torch.no_grad():
        main()