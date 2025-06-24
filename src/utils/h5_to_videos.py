import h5py
import numpy as np
import cv2
import sys
from config import data_path
def h5_to_video(h5_path, output_path, fps=30):
    with h5py.File(h5_path, "r") as f:
        frames = f["vids"][:]
        print(f"Loaded {frames.shape[0]} frames of shape {frames.shape[1:]}")
    height, width = frames.shape[1], frames.shape[2]
    if frames.ndim == 4:
        channels = frames.shape[3]
    else:
        channels = 1
        frames = np.expand_dims(frames, -1)

    # OpenCV expects BGR for color, grayscale otherwise
    if channels == 1:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        for frame in frames:
            frame_uint8 = (frame.squeeze() * 255).astype(np.uint8) if frame.max() <= 1 else frame.squeeze().astype(np.uint8)
            
            video_writer.write(cv2.cvtColor(frame_uint8, cv2.COLOR_GRAY2BGR))
    else:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        for frame in frames:
            frame_uint8 = (frame * 255).astype(np.uint8) if frame.max() <= 1 else frame.astype(np.uint8)
            video_writer.write(frame_uint8)
    video_writer.release()
    print(f"Video saved to {output_path}")

if __name__ == "__main__":
    import glob 
    h5paths = glob.glob(f"{data_path}/**/vid*.h5", recursive=True)
    for h5_path in h5paths:
        print(f"Processing {h5_path}")
        output_path = h5_path.replace(".h5", ".mp4")
        fps = 30
        h5_to_video(h5_path, output_path, fps)