import sys
sys.path.append("/usr/lib/python3/dist-packages")
from metavision_sdk_stream import Camera, CameraStreamSlicer, SliceCondition
import numpy as np
import cv2
import torch

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

    if args.input_event_file:
        camera = Camera.from_file(args.input_event_file)
    else:
        camera = Camera.from_first_available()

    width, height = camera.width(), camera.height()
    print(f"Camera resolution: {width}x{height}")
    slice_condition = SliceCondition.make_n_us(10000)
    slicer = CameraStreamSlicer(camera.move(), slice_condition=slice_condition)

    for slice in slicer:
        if slice.events.size == 0:
            print("The current event slice is empty.")
            continue

        # Create a black image
        img = np.zeros((height, width, 3), dtype=np.uint8)

        # Draw events: polarity 1 as white, 0 as blue (or any color you like)
        xs = slice.events['x']
        ys = slice.events['y']
        ps = slice.events['p'] if 'p' in slice.events.dtype.names else np.ones_like(xs)

        img[ys[ps == 1], xs[ps == 1]] = (255, 255, 255)  # White for polarity 1
        img[ys[ps == 0], xs[ps == 0]] = (255, 0, 0)      # Blue for polarity 0

        cv2.imshow("Event Frame", img)
        key = cv2.waitKey(1)
        if key == 27:  # ESC to quit
            break

        

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()