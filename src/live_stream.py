from models.BOBW import BestOfBothWorld
from config import checkpoint_path

import cv2
import torch


## check if current pythin version is 3.12
import sys
if sys.version_info.major != 3 or sys.version_info.minor < 12:
    from utils.dataviewers import dataviewer39 as dataviewer
else:
    from utils.dataviewers import dataviewer312 as dataviewer
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
    network = "BOBWLSTM"
    viewer = dataviewer()
    
    if checkpoint_path:
        checkpoint_file = f'{checkpoint_path}/model_epoch_5_BOBWLSTM_small.pth'
        if "small" in checkpoint_file:
            model = BestOfBothWorld(model_type=network, width=viewer.width, height = viewer.height,embed_dim=128, depth=6, heads=8, num_queries=16)
        else:
            model = BestOfBothWorld(model_type=network, width=viewer.width, height = viewer.height,embed_dim=256, depth=12, heads=8, num_queries=64)
        print(f"Loading checkpoint from {checkpoint_file}")
        try:
            model.load_state_dict(torch.load(checkpoint_file, map_location=device))
            epoch_checkpoint = int(checkpoint_file.split("_")[2]) + 1
        except:
            print("Checkpoint not found, starting from scratch")
    model.to(device)
    viewer.setModel(model)
    viewer.run()
    
    
        

        

    cv2.destroyAllWindows()

if __name__ == "__main__":
    with torch.no_grad():
        main()