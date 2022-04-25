import os
import time
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
from tensorflow.keras.utils import Progbar

from utils.utils import load_lines, json2model, load_weights, compute_markers, non_max_suppression
import utils.data_utils as data_utils
from options import get_options


# Constants
TEST_PHASE = 0
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def main(opts):
    K.clear_session()

    # Load file containing the path of the training images
    lines_test = load_lines(opts.test_imgs, opts.dataset_path)

    # Reduce the number of classifiers
    filter_classifiers = opts.dataset_path + opts.filter_img if opts.filter_classifiers else None

    # Generate testing data
    test_generator = data_utils.DataGenerator(lines_test,
                                              opts.num_classes,
                                              opts.v_grid,
                                              opts.h_grid,
                                              test=True,
                                              img_mode=opts.img_mode,
                                              target_size=(opts.img_height, opts.img_width),
                                              batch_size=opts.batch_size_test,
                                              filter_img=filter_classifiers,
                                              max_boxes=opts.max_boxes)

    # Load json and create model
    model = json2model(os.path.join(opts.save_dir, opts.json_model))

    # Load weights
    model = load_weights(model, os.path.join(opts.save_dir, opts.weights))

    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam')

    # Define grid steps
    dx = test_generator.dx
    dy = test_generator.dy

    # Define grid positions
    grid = test_generator.grid

    # Progress bar
    progbar = Progbar(target=test_generator.samples)
    steps_done = 0

    # Create folders to save outputs (for mAP calculation)
    model_dir = opts.save_dir.split('/')[-1] + '_test_{}'.format(time.strftime("%Y%m%dT%H%M%S"))
    det_dir = os.path.join('./map/predictions/', model_dir, 'detection-results')
    gt_dir = os.path.join('./map/predictions/', model_dir, 'ground-truth')
    img_dir = os.path.join('./map/predictions/', model_dir, 'images')
    if not os.path.exists(det_dir):
        os.makedirs(det_dir)
    if not os.path.exists(gt_dir):
        os.makedirs(gt_dir)
    if not os.path.exists(img_dir) and opts.save_imgs:
        os.makedirs(img_dir)

    # Iterate over images
    for batches in test_generator:
        pred = model.predict_on_batch(batches[0])
        pred = np.array(pred).squeeze()

        for n in range(batches[0].shape[0]):

            gt = np.array(batches[1][n])
            gt[:, [1, 0]] = gt[:, [0, 1]]
            gt = gt[np.logical_and(gt[:, 0] > 0, gt[:, 1] > 0)]
            path = test_generator.batch_lines[n].split(' ')[0]
            sequence = path.split('/')[-2]

            # Get the predicted probabilities
            if test_generator.batch_size == 1:
                prob = pred
            else:
                prob = pred[n, :]

            # Calculate the positions and scores of the markers
            markers = compute_markers(prob, grid, opts.m_th, dx, dy)

            # Visualization mode (slower)
            if opts.visualize or opts.save_imgs:

                # Load the image not resized
                img = data_utils.load_img(path)

                # Variables to extrapolate coordinates to the real size of the image
                ly = img.shape[0] / opts.img_height
                lx = img.shape[1] / opts.img_width

            # If at least 1 marker is predicted, apply NMS
            if len(markers) > 0:

                # Apply NMS
                markers = non_max_suppression(markers, opts.d_th)

            # Save predictions for mAP calculation
            results = open(os.path.join(det_dir,
                                        path.replace("/", "_").replace('.jpg', ".txt").replace('.png', ".txt")), 'w')
            for marker in markers:
                results.write('vehicle {} {} {}\n'.format(marker[2], marker[0], marker[1]))
            results.close()

            # Save gt for mAP calculation
            results = open(os.path.join(gt_dir,
                                        path.replace("/", "_").replace('.jpg', ".txt").replace('.png', ".txt")), 'w')
            for g in gt:
                results.write('vehicle {} {}\n'.format(g[0], g[1]))
            results.close()

            # Visualization mode (slower)
            if opts.visualize or opts.save_imgs:

                # Show image
                plt.imshow(img)
                plt.axis('off')

                # Plot grid
                if opts.view_grid:
                    plt.scatter(grid[:, 1] * lx, grid[:, 0] * ly, marker='o', c=prob, cmap='spring', alpha=0.25)
                    # plt.colorbar()

                # Extrapolate the coordinates to the real size of the image
                if len(markers) > 0:
                    markers[:, 0] = markers[:, 0] * ly
                    markers[:, 1] = markers[:, 1] * lx
                gt[:, 0] = gt[:, 0] * ly
                gt[:, 1] = gt[:, 1] * lx

                # Plot gt
                for g in gt:
                    plt.scatter(g[1], g[0], c='red', marker='o', alpha=0.5)

                # Plot markers
                for marker in markers:
                    plt.annotate(str(round(marker[2], 2)), xy=(marker[1], marker[0]),
                                 xytext=(marker[1] + 20, marker[0] + 20), xycoords='data',
                                 bbox=dict(boxstyle='round,pad=0.2', fc='green', alpha=0.25))
                    plt.scatter(marker[1], marker[0], c='green', marker='o', alpha=0.5)

                # Save images
                if opts.save_imgs:
                    p = os.path.join(img_dir, sequence)
                    if not os.path.exists(p):
                        os.makedirs(p)
                    s = path.split('/')[-1]
                    plt.savefig(os.path.join(p, s))

                # Show image and clean figure
                if opts.visualize:
                    plt.show()
                plt.clf()

            # Update progress bar
            steps_done += 1
            progbar.update(steps_done)
            if steps_done >= test_generator.samples:
                break
        if steps_done >= test_generator.samples:
            break
    print('Finished')


if __name__ == "__main__":
    main(get_options())
