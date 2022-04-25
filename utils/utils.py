import os
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import model_from_json


def load_lines(path_to_lines, dataset_path, shuffle=True):
    """
    Load lines in path_to_lines, add to them the absolute path, and (maybe) shuffle them.
    """
    with open(path_to_lines) as f:
        lines = f.readlines()
    for i in range(len(lines)):
        lines[i] = os.path.join(dataset_path, lines[i].replace('\n', ''))
    if shuffle:
        np.random.shuffle(lines)
    return lines


def load_weights(model, path):
    """
    Load weights in path on model.
    """
    if path is not None and os.path.isfile(path):
        try:
            model.load_weights(path)
            print("Loaded model from {}".format(path))
        except:
            print("Impossible to find weight path. Returning untrained model")
    return model


def model2json(model, json_model_path):
    """
    Serialize model into json.
    """
    model_json = model.to_json()

    with open(json_model_path, "w") as f:
        f.write(model_json)


def json2model(json_model_path):
    """
    Serialize json into model.
    """
    with open(json_model_path, 'r') as json_file:
        loaded_model_json = json_file.read()

    model = model_from_json(loaded_model_json)
    return model


def plot_loss(path_to_log):
    """
    Read log file and plot losses.

    # Arguments
        path_to_log: Path to log file.
    """
    # Read log file
    log_file = os.path.join(path_to_log, "log.txt")
    try:
        log = np.genfromtxt(log_file, delimiter='\t', dtype=None, names=True)
    except:
        raise IOError("Log file not found")

    train_loss = log['train_loss']
    val_loss = log['val_loss']
    timesteps = list(range(train_loss.shape[0]))

    # Plot losses
    plt.plot(timesteps, train_loss, 'r--', timesteps, val_loss, 'b--')
    plt.legend(["Training loss", "Validation loss"])
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.savefig(os.path.join(path_to_log, "log.png"))
    # plt.show()


def compute_markers(p, grid, t, dx, dy):
    """
    Compute the positions and scores of the markers with the probability of each classifier predicted by the model. The
    scores are based on the probability of the marker, which is calculated by interpolation. For the markers of the gt,
    it is not needed to interpolate, just get the middle position.

    # Arguments
        p: probability of each classifier. For the markers of the gt, p = gt
        grid: coordinates of each classifier (must have the same length as p)
        t: probabilities higher than this threshold are marked as positive detections
        dx: grid step in x-axis
        dx: grid step in y-axis

    # Returns
        markers: array with the coordinates and score of each marker. For the markers of the gt, just the coordinates
        are saved.
    """

    p = np.array(p)
    markers = []

    # Relative coordinates of neighbors
    neighbors = np.array([[0, dy], [dx, 0], [dx, dy]])

    # Iterate over each classifier
    classifiers = np.concatenate((grid, np.expand_dims(p, 1)), axis=1)
    for classifier in classifiers:

        # Get coordinates and prob value of each neighbor
        n = [classifier]
        for neighbor in neighbors:
            coords = np.add(classifier[:2], neighbor)
            d = compute_distance(coords, grid)
            if np.min(d) < 1e-15:
                idx = np.argmin(d)
                n.append([coords[0], coords[1], p[idx]])
        n = np.array(n)

        # Check if threshold is surpassed
        prob = np.sum(n[:, 2])
        if prob >= t * n.shape[0]:  # and np.all(n[:, 2] >= t / 10)

            # Calculate the centroid
            absolute = n[0, :2].copy()
            n[:, :2] = n[:, :2] - absolute
            cy = np.sum(n[:, 0] * n[:, 2]) / prob  # Weighted interpolation
            cx = np.sum(n[:, 1] * n[:, 2]) / prob
            # cy = (np.max(n[:, 0]) - np.min(n[:, 0])) / 2  # Center of the classifiers
            # cx = (np.max(n[:, 1]) - np.min(n[:, 1])) / 2

            # Append the coordinates and the prob value of the new classifier
            markers.append([cy + absolute[0], cx + absolute[1], prob / n.shape[0]])

    return np.array(markers)


def non_max_suppression(markers, threshold):
    """
    Performs non-maximum suppression.

    # Arguments
        markers: 2-D array with 1 where there are markers.
        scores: 2-D array with the scores of the markers.
        threshold: distance threshold to use for filtering.

    # Returns
        output: array with the positions of the markers not filtered.
    """

    # Initialize the output of the function
    output = []

    # Get the coordinates of the markers
    coordinates = markers[:, :2]

    # Get the scores of the markers and save them in 1-D array
    scores = markers[:, 2]

    # Get indexes of markers sorted by scores (highest first)
    ixs = scores.argsort()[::-1]

    # Initialize array that will save the markers not filtered
    pick = []

    # Iterate over the list of indexes
    while len(ixs) > 0:
        # Pick marker with highest score and add its index to the list "pick"
        i = ixs[0]
        pick.append(i)

        # Compute distance between the picked marker and the rest
        d = compute_distance(coordinates[i], coordinates[ixs[1:]])

        # Identify markers with distance over the threshold
        # This returns indexes into ixs[1:], so add 1 to get indexes into ixs
        remove_ixs = np.where(d <= threshold)[0] + 1

        # Remove indexes of the picked and overlapped markers.
        ixs = np.delete(ixs, remove_ixs[remove_ixs < len(ixs)])
        ixs = np.delete(ixs, 0)

    # Iterate over the picked markers and put 1 in their positions inside the array "output"
    for c in range(np.size(coordinates[pick], axis=0)):
        output.append([coordinates[pick][c, 0], coordinates[pick][c, 1], scores[pick][c]])

    return np.array(output)


def compute_distance(coords, ref_coords):
    """
    Calculates distance between the coordinates of reference and the others coordinates.

    # Arguments
        coords: 1D vector [y, x]
        ref_coords: list with the others coordinates

    # Returns
        Distance between the coordinates of reference and the others coordinates.
    """

    x1 = coords[0]
    y1 = coords[1]

    ref_coords = np.array(ref_coords)
    x2 = ref_coords[..., 0]
    y2 = ref_coords[..., 1]

    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


class HistoryCallback(Callback):

    def __init__(self, path):
        super().__init__()
        self.path = path

    def on_epoch_end(self, epoch, logs=None):
        if os.path.isfile(os.path.join(self.path, 'history.pkl')):
            with open(os.path.join(self.path, 'history.pkl'), 'rb') as f:
                history = pickle.load(f)
            history[epoch] = logs
        else:
            history = {epoch: logs}
        with open(os.path.join(self.path, 'history.pkl'), 'wb') as f:
            pickle.dump(history, f)


class WeightedBinaryCrossEntropy(tf.keras.losses.Loss):
    """Weighted Binary Cross-Entropy loss function"""

    def __init__(self, zero_weight=1, one_weight=1):
        super().__init__()
        self.zero_weight = zero_weight
        self.one_weight = one_weight

    def call(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        b_ce = K.binary_crossentropy(y_true, y_pred)
        weight_vector = y_true * self.one_weight + (1 - y_true) * self.zero_weight
        weighted_b_ce = weight_vector * b_ce
        return K.mean(weighted_b_ce, axis=-1)
