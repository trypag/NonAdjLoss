import argparse
import cv2
import matplotlib.pyplot
import numpy as np
import os
import torch

from torchmed.patterns import SquaredSlidingWindow
from torchmed.readers import SitkReader


parser = argparse.ArgumentParser(
    description='Script to extract the adjacency matrix from a segmentation dataset')
parser.add_argument('data', metavar='SOURCE_DIR', help='path to the source dataset')
parser.add_argument('output', metavar='DEST_DIR', help='path to the destination directory')
parser.add_argument('nb_labels', default=135, metavar='N_LABELS', type=int,
                    help='number of labels in the dataset')
parser.add_argument('--n-size', default=1, type=int, metavar='SIZE',
                    help='size of the neighborhood')


def main():
    args = parser.parse_args()
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    total_adj_mat = torch.FloatTensor(args.nb_labels, args.nb_labels).zero_()

    for train_id in os.listdir(args.data):
        train_patient = os.path.join(args.data, train_id)
        if os.path.isdir(train_patient) and train_id.isdigit():

            # adjacency matrix with one additional dimension for discarded label (-1)
            img_adj = torch.FloatTensor(args.nb_labels + 1, args.nb_labels + 1).zero_()
            # reads a segmentation map
            label = SitkReader(train_patient + '/prepro_seg_mni.nii.gz')

            # image array from the reader
            label_array = label.to_torch().long()
            # re-label discarded label (-1) by the last positive integer
            label_array[label_array == -1] = args.nb_labels

            # extract adjacency matrix from the image and fill in the matrix
            image2graph3d_patch(label_array, img_adj.numpy(), args.nb_labels, args.n_size)
            # discard last positive label (discarded label, you remember ?)
            img_adj = img_adj[:-1, :-1]
            # sum with the global adjacency matrix
            total_adj_mat += img_adj

    save2png(total_adj_mat, os.path.join(args.output, 'adjacency_n_' + str(args.n_size)))


def image2graph3d_patch(image, adj_mat, n_classes, n_size=1):
    """
    Offline function : operates on raw volumes
    Converts 3D image to an adjacency matrix. Operates by sliding a 3D patch.
    Counts neighborhood for each pixels.
    """
    # 3D patch size based on neighborhood size (n_size)
    patch_size = (2 * n_size + 1,) * 3
    # sliding window with padding of value n_classes - 1 on borders
    pattern = SquaredSlidingWindow(patch_size, True, n_classes)
    pattern.prepare(image)

    # loop over each voxel of the image
    for x in range(image.size(0)):
        for y in range(image.size(1)):
            for z in range(image.size(2)):
                # extract the 3D patch
                t = pattern(image, (x, y, z)).contiguous()
                center = t[n_size, n_size, n_size]

                # add one to the corresponding indices of the center
                np.add.at(adj_mat[center], t.view(-1).numpy(), 1)
                adj_mat[center, center] -= 1


def save2png(image, output):
    # save to file
    np.savetxt(output + '.csv', image.float().numpy(), delimiter=';')

    # save binary image
    brain = (image > 0).float().numpy()
    brain = cv2.cvtColor(brain, cv2.COLOR_GRAY2BGR)
    brain = cv2.normalize(brain, brain, 0, 1, cv2.NORM_MINMAX)
    matplotlib.pyplot.imsave(output + '.png', brain)


if __name__ == '__main__':
    main()
