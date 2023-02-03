"""
This script visualizes a recording by displaying all cameras/depth/IR in a single video
"""
import os
import cv2
import argparse
import glob
import random
from recording.sequence_reader import SequenceReader
from tqdm import tqdm


def vis_data(seq_path):
    print('Viewing', seq_path)
    seq_reader = SequenceReader(seq_path)
    disp_frames = []

    # for i in tqdm(range(seq_reader.num_frames)):
    #     frame = seq_reader.get_overall_frame(i, overlay_force=True)
    #     disp_frames.append(frame)

    t = 0
    while True:
        if len(disp_frames) > t:
            # print('Fetching old frame')
            frame = disp_frames[t]
        else:
            # print('Getting new frame')
            frame = seq_reader.get_overall_frame(t, overlay_force=True)
            disp_frames.append(frame)

        # frame = frame.copy()
        cv2.putText(frame, str(t), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (1, 1, 1))
        cv2.imshow('frame', frame)
        t = (t + 1) % len(seq_reader.timesteps)

        keycode = cv2.waitKey(10) & 0xFF
        if keycode == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str, default='')
    args = parser.parse_args()

    if len(args.path) == 0:
        vis_data(random.choice(glob.glob('data/test/*/*')))
    else:
        vis_data(args.path)

