import cv2
import math
import PIL.Image
import numpy as np
from tensorflow import keras
from PIL import Image, ImageEnhance
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb


class DataGenerator(keras.utils.Sequence):
    """Generates data for Keras"""
    def __init__(self, lines, output_dim, v_grid, h_grid, batch_size=32, target_size=(224, 224), filter_img=None,
                 img_mode='grayscale', data_aug=False, test=False, shuffle=False, max_boxes=20):
        """Initialization"""

        # Files
        self.lines = lines
        self.samples = len(lines)

        # Images
        self.target_size = tuple(target_size)
        self.batch_size = batch_size

        # Number of classes
        self.output_dim = output_dim
        # Data augmentation
        self.data_aug = data_aug

        # Initialize image mode
        if img_mode not in {'rgb', 'grayscale'}:
            raise ValueError('Invalid color mode:', img_mode,
                             '; expected "rgb" or "grayscale".')
        self.img_mode = img_mode
        self.grayscale = self.img_mode == 'grayscale'
        if self.grayscale:
            self.image_shape = self.target_size + (1,)
        else:
            self.image_shape = self.target_size + (3,)

        # Ground truth for GSAC
        self.dx = round(target_size[1] / h_grid)
        self.dy = round(target_size[0] / v_grid)
        grid_x = np.arange(round(self.dx / 2), target_size[1] - 1, self.dx).astype(int)
        grid_y = np.arange(round(self.dy / 2), target_size[0] - 1, self.dy).astype(int)
        if filter_img is not None:
            mask = np.squeeze(load_img(filter_img, target_size=target_size, grayscale=True))
            aux = np.zeros(target_size)
            for gx in grid_x:
                for gy in grid_y:
                    aux[gy, gx] = 1
            aux = np.logical_and(aux, mask)
            self.grid = np.argwhere(aux == 1)
        else:
            grid = []
            for gx in grid_x:
                for gy in grid_y:
                    grid.append([gx, gy])
            self.grid = np.array(grid)

        # Other info
        self.test = test
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.lines))
        self.max_boxes = max_boxes
        self.paths = []
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.lines) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""

        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]

        # Find list of IDs
        self.batch_lines = [self.lines[k] for k in indexes]

        # Generate data
        x, y = self.__data_generation(self.batch_lines)

        return x, y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.lines))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, lines):
        """Generates data containing batch_size samples"""

        # Initialization
        batch_x = np.empty((self.batch_size, *self.image_shape))  # X : (n_samples, *dim, n_channels)
        if self.test:
            batch_y = np.zeros((self.batch_size, self.max_boxes, 2), dtype=float)
        else:
            batch_y = np.zeros((self.batch_size, self.grid.shape[0]), dtype=int)

        # Generate data
        for i, line in enumerate(lines):
            line_split = line.split(' ')

            # Load image
            x, true_shape = load_img(line_split[0], grayscale=self.grayscale, target_size=self.target_size,
                                     original_shape=True)

            # Scale factor if the images are resized
            scale_x = true_shape[1] / self.target_size[1]
            scale_y = true_shape[0] / self.target_size[0]

            # Get the gt of the chosen image
            boxes = []
            for box in range(1, len(line_split)):
                left = float(line_split[box].split(',')[0]) / scale_x
                top = float(line_split[box].split(',')[1]) / scale_y
                boxes.append([left, top])

            # Data augmentation
            if self.data_aug:
                x, boxes = data_augmentation(x, boxes)

            # Save image
            batch_x[i] = x

            # Save the gts
            for j, box in enumerate(boxes):
                left, top = box[0], box[1]
                if self.test:
                    assert j >= self.max_boxes, \
                        'Image {} has more bounding boxes than the maximum ({}). You should consider a higher value ' \
                        'for the option max_boxes (in options.py)'.format(line_split[0], self.max_boxes)
                    batch_y[i, j, :] = [left, top]
                else:
                    batch_y[i] = np.logical_or(batch_y[i], gt_reform(self.dx, self.dy, self.grid, left, top))
        return batch_x, batch_y


def load_img(path, grayscale=False, target_size=None, original_shape=False):

    """
    Load an image.

    # Arguments
        path: Path to image file.
        grayscale: Boolean, whether to load the image as grayscale.
        target_size: Either `None` (default to original size)
            or tuple of ints `(img_height, img_width)`.
        original_shape: whether to output the original shape or not.

    # Returns
        Image as numpy array.
    """

    # Read input image
    img = cv2.imread(path)
    shape = img.shape

    if grayscale:
        if len(img.shape) != 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if target_size:
        if (img.shape[0], img.shape[1]) != target_size:
            img = cv2.resize(img, (target_size[1], target_size[0]))

    if grayscale:
        img = img.reshape((img.shape[0], img.shape[1], 1))

    if original_shape:
        return np.asarray(img, dtype=np.float64) / 255, shape
    return np.asarray(img, dtype=np.float64) / 255


def data_augmentation(image, boxes, use_bb=False):

    image = Image.fromarray(np.uint8(image * 255)).convert('RGB')
    w, h = image.size
    boxes = boxes[~np.all(boxes == 0, axis=1)]

    # Resize
    if rand() > 1:
        jitter = 0.3
        min_scale_factor = 0.75
        max_scale_factor = 1
        new_ar = w / h * rand(1 - jitter, 1 + jitter) / rand(1 - jitter, 1 + jitter)
        scale = rand(min_scale_factor, max_scale_factor)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)
        dx = int(rand(0, w - nw))
        dy = int(rand(0, h - nh))
        new_image = Image.new('RGB', (w, h), (0, 0, 0))
        new_image.paste(image, (dx, dy))
        image = new_image
        if use_bb:
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * nw / w + dx
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * nh / h + dy
        else:
            boxes[:, 0] = boxes[:, 0] * nw / w + dx
            boxes[:, 1] = boxes[:, 1] * nh / h + dy

    # Horizontal Flip
    if rand() > 0.5:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        if use_bb:
            boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
        else:
            boxes[:, 0] = w - boxes[:, 0]

    # Vertical Flip
    if rand() > 0.5:
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        if use_bb:
            boxes[:, [1, 3]] = h - boxes[:, [3, 1]]
        else:
            boxes[:, 1] = h - boxes[:, 1]

    # Rotate
    if rand() > 1 and not use_bb:
        angle = rand(0, 90)
        image = image.rotate(angle, PIL.Image.NEAREST, expand=1)
        boxes = rotate((w // 2, h // 2), boxes, math.radians(angle))

    # Brightness change
    if rand() > 0.5:
        factor = rand(0.5, 1.5)
        image = ImageEnhance.Brightness(image).enhance(factor)

    # Contrast change
    if rand() > 0.5:
        factor = rand(0.5, 1.5)
        image = ImageEnhance.Contrast(image).enhance(factor)

    image = np.array(image) / 255

    # HSV Transform
    if rand() > 1:  # Deactivated
        hue = 0.1
        sat = 1.5
        val = 1.5
        hue = rand(-hue, hue)
        sat = rand(1, sat) if rand() < .5 else 1 / rand(1, sat)
        val = rand(1, val) if rand() < .5 else 1 / rand(1, val)
        x = rgb_to_hsv(image)
        x[..., 0] += hue
        x[..., 0][x[..., 0] > 1] -= 1
        x[..., 0][x[..., 0] < 0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x > 1] = 1
        x[x < 0] = 0
        image = hsv_to_rgb(x)  # numpy array, 0 to 1

    # Shear
    if rand() > 1:  # Deactivated
        shear_factor = 0.2
        shear_factor = rand(-shear_factor, shear_factor)
        M = np.array([[1, abs(shear_factor), 0], [0, 1, 0]])
        nW = w + abs(shear_factor * h)
        image = cv2.warpAffine(image, M, (int(nW), h))
        image = cv2.resize(image, (w, h))
        scale_factor_x = nW / w
        if use_bb:
            boxes[:, [0, 2]] += ((boxes[:, [1, 3]]) * abs(shear_factor)).astype(int)
            boxes[:, :4] = boxes[:, :4] / [scale_factor_x, 1, scale_factor_x, 1]
        else:
            boxes[:, 0] = boxes[:, 0] / (1 + (scale_factor_x - 1) / 2)

    # Amend boxes
    boxes[:, 0:2][boxes[:, 0:2] < 0] = 0
    if use_bb:
        boxes[:, 2][boxes[:, 2] > w] = w
        boxes[:, 3][boxes[:, 3] > h] = h
        box_w = boxes[:, 2] - boxes[:, 0]
        box_h = boxes[:, 3] - boxes[:, 1]
        boxes = boxes[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid box

        min_box_area = 200
        boxes = boxes[bbox_area(boxes) > min_box_area]
    else:
        boxes[:, 0][boxes[:, 0] > w] = w
        boxes[:, 1][boxes[:, 1] > h] = h

    return image, boxes


def bbox_area(bbox):
    return (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1])


def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy


def rand(a=0., b=1.):
    return np.random.rand()*(b-a) + a


def gt_reform(dx, dy, grid, x, y):

    """
    Transform the format x|y|h|w of the gt to a binary vector where '1' means that a car is present on the corresponding
    grid point and '0' means that there is no car.

    # Arguments
        dx: grid step in x-axis
        dy: grid step in y-axis
        grid: grid positions
        x: position in x axis of the gt
        y: position in y axis of the gt

    # Returns
        Binary vector ordered by the columns of the grid that indicates with 1.
    """

    # Initialize output gt
    gt = np.zeros(grid.shape[0])

    # Avoid checking all the grid if there are not markers
    if (x <= 0) | (y <= 0):
        return gt

    # Create square around the marker
    x1 = x - dx
    x2 = x + dx
    y1 = y - dy
    y2 = y + dy

    # Iterate over grids
    for i in range(grid.shape[0]):
        if (x1 < grid[i, 1] <= x2) and (y1 < grid[i, 0] <= y2):
            gt[i] = 1
    return gt
