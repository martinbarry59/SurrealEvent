from models.BOBW import BestOfBothWorld
from config import checkpoint_path
import sys
sys.path.append("/usr/lib/python3/dist-packages")
from metavision_sdk_stream import Camera, CameraStreamSlicer, SliceCondition
import numpy as np
import cv2
import torch
from metavision_sdk_cv import ActivityNoiseFilterAlgorithm
from metavision_sdk_base import EventCDBuffer

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
    
    if checkpoint_path:
        checkpoint_file = f'{checkpoint_path}/model_epoch_5_BOBWLSTM.pth'
        if "small" in checkpoint_file:
            model = BestOfBothWorld(model_type=network, width=320, height = 320,embed_dim=128, depth=6, heads=8, num_queries=16)
        else:
            model = BestOfBothWorld(model_type=network, width=320, height = 320,embed_dim=256, depth=12, heads=8, num_queries=64)
        print(f"Loading checkpoint from {checkpoint_file}")
        model.load_state_dict(torch.load(checkpoint_file, map_location=device))
        try:
            model.load_state_dict(torch.load(checkpoint_file, map_location=device))
            epoch_checkpoint = int(checkpoint_file.split("_")[2]) + 1
        except:
            print("Checkpoint not found, starting from scratch")
    model.to(device)
    if args.input_event_file:
        camera = Camera.from_file(args.input_event_file)
    else:
        camera = Camera.from_first_available()

    width, height = camera.width(), camera.height()
    print(f"Camera resolution: {width}x{height}")
    slice_condition = SliceCondition.make_n_us(100000)
    slicer = CameraStreamSlicer(camera.move(), slice_condition=slice_condition)
    activity_filter = ActivityNoiseFilterAlgorithm(width, height, 5000)

    for slice in slicer:
        if slice.events.size == 0:
            print("The current event slice is empty.")
            continue
        events_buf = EventCDBuffer()

        # In your event processing loop:
        activity_filter.process_events(slice.events, events_buf)
        filtered_events = events_buf.numpy().copy()
        events = filtered_events
        # Create a black image
        img = np.zeros((height, width, 3), dtype=np.uint8)
        N = 2000
        # Draw events: polarity 1 as white, 0 as blue (or any color you like)
        xs = events[:N]['x']
        ys = events[:N]['y']
        ps = events[:N]['p']* 2 - 1
        ts = events[:N]['t']

        events_tensor = torch.stack(( torch.tensor(ts).float(), torch.tensor(xs / 320).float(),
                                      torch.tensor(ys / 320).float(), 
                                     torch.tensor(ps).float()), dim=1)  # [N, 4].long()
        events_tensor = events_tensor
        mask = torch.ones((events_tensor.shape[0],), dtype=torch.bool)
        seq_events = events_tensor.unsqueeze(0).unsqueeze(0).to(device)
        seq_masks = mask.unsqueeze(0).unsqueeze(0).to(device)
        predictions, encodings = model(seq_events, seq_masks)
        img[ys[ps == 1], xs[ps == 1]] = (255, 255, 255)  # White for polarity 1
        img[ys[ps == 0], xs[ps == 0]] = (255, 0, 0)      # Blue for polarity 0

        ## crop image to 246, 346
        # merge the predictions into the image
        if predictions is not None:
            pred = predictions[0, 0].detach().cpu().numpy()
            pred = (pred * 255).astype(np.uint8)
            pred = cv2.applyColorMap(pred, cv2.COLORMAP_JET)
            img = cv2.addWeighted(img, 0.5, pred, 0.5, 0)
        cv2.imshow("Event Frame", img)
        key = cv2.waitKey(1)
        if key == 27:  # ESC to quit
            break

        

    cv2.destroyAllWindows()

if __name__ == "__main__":
    with torch.no_grad():
        main()