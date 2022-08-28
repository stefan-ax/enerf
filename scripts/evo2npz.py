import json
import os
import argparse
import numpy as np
from scipy.spatial.transform import Rotation
from tqdm.auto import tqdm
from random import sample


# Data source: https://rpg.ifi.uzh.ch/davis_data.html (IJRR)

def quat2rot(quat: list):
    rot = Rotation.from_quat(quat)
    return rot.as_matrix()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Utility to transform event data to Tiny_NeRF format')
    parser.add_argument('path', help='Path to calib.txt, groundtruth.txt, events.txt files')
    parser.add_argument('--N', help='Fixed number of poses to be sampled', type=int)
    args = parser.parse_args()

    assert os.path.isdir(args.path)

    print("Processing poses")
    with open(os.path.join(args.path, 'groundtruth.txt')) as fin:
        lines = fin.readlines()

        if args.N:
            N = args.N
            step = len(lines) // N
        else:
            N = len(lines) - 1
            step = 1

        poses = np.zeros((N, 2, 4, 4))
        timestamps = np.zeros((N, 2))

        for idx, idx_lines in zip(range(N), range(0, len(lines) - 1, step)):
            line = lines[idx_lines]
            next_line = lines[idx_lines + 1]

            t, px, py, pz, qx, qy, qz, qw = [float(el) for el in line.split(" ")]
            timestamps[idx, 0] = t
            poses[idx, 0, :, -1] = [px, py, pz, 1]
            poses[idx, 0, :3, :3] = quat2rot([qx, qy, qz, qw])

            t, px, py, pz, qx, qy, qz, qw = [float(el) for el in next_line.split(" ")]
            timestamps[idx, 1] = t
            poses[idx, 1, :, -1] = [px, py, pz, 1]
            poses[idx, 1, :3, :3] = quat2rot([qx, qy, qz, qw])

        # poses final shape: [N, 2, 4, 4]
        # timestamps final shape: [N, 2]

    print('Processing events')
    with open(os.path.join(args.path, 'events.txt')) as fin:
        images = np.zeros((N, 240, 180))  # final shape: [N, 240, 180]

        general_idx = 0

        for line in tqdm(fin.readlines()):
            t, x, y, p = [float(el) for el in line.split(" ")]
            if p == 0: p = -1
            if t < timestamps[0, 0] or t > timestamps[-1, 1]:  # Event before/after pose recording
                continue

            if t <= timestamps[general_idx + 1, 0]:
                images[general_idx, int(x), int(y)] += p
            else:
                try:
                    while t > timestamps[general_idx + 1, 0]:
                        general_idx += 1
                except IndexError:
                    break

                images[general_idx, int(x), int(y)] += p

        print(f"Images rescaled by dividing of maximum {images.max()}")
        images = images / images.max()


    with open(os.path.join(args.path, 'calib.txt')) as fin:
        focal = float(fin.read().split(" ")[0])

    basename = os.path.basename(args.path)
    np.savez(f'data/{basename}.npz', **{
        'images': images,
        'poses': poses,
        'focal': focal,
        'timestamps': timestamps
    })

    print('Done')
