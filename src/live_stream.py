from models.ConvLSTM import EConvlstm
from config import checkpoint_path

import cv2
import torch


## check if current pythin version is 3.12
import sys
## get connected camera names
try: 
    sys.path.append("/usr/lib/python3/dist-packages")

    from metavision_sdk_stream import Camera
    camera = Camera.from_first_available()
    from utils.dataviewers import dataviewerprophesee as dataviewer

except Exception as e:
    try:
        import dv_processing as dv
        camera = dv.io.camera.open()
        from utils.dataviewers import dataviewerdavis as dataviewer
    except Exception as e:
        print("could not find any cameras" )
    
        exit()

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
    network = "CONVLSTM"
    viewer = dataviewer(camera)
    
    if checkpoint_path:
        checkpoint_file = f'{checkpoint_path}/model_epoch_1_CONVLSTM.pth'
        # checkpoint_file = f'{checkpoint_path}/model_epoch_8_CONVLSTM_best_SKIP_NOLSTM.pth'
        if "NOLSTM" in checkpoint_file:
            model = EConvlstm(model_type=network, skip_lstm=False)
        else:
            model = EConvlstm(model_type=network, skip_lstm=True)
        print(f"Loading checkpoint from {checkpoint_file}")
        try:
            model.load_state_dict(torch.load(checkpoint_file, map_location=device))
        except:
            print("Checkpoint not found, starting from scratch")
    model.to(device)
    viewer.setModel(model)
    viewer.run()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    with torch.no_grad():
        main()