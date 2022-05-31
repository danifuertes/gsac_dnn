import os
import time
import pickle
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, TensorBoard

from utils.utils import load_lines, model2json, load_weights, WeightedBinaryCrossEntropy, HistoryCallback
from resnet_models import network, lr_schedule
import utils.data_utils as data_utils
from options import get_options


# Constants
TRAIN_PHASE = 1
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def create_model(n, version, classifier_conf, img_width, img_height, img_channels, num_classes, num_classifiers,
                 weights_path, resnet_keras_weights):
    """
    Create GSAC-DNN model.

    # Arguments
       n: parameter that determines the net depth.
       version: 1 for resnet v1, 2 for v2 or keras for the version of Keras.
       classifier_conf: ID to identify the architecture of the classifiers.
       img_width: Target image widht.
       img_height: Target image height.
       img_channels: Target image channels.
       num_classes: Dimension of model output (number of classes).
       num_classifiers: Number of classifiers based on grid size.
       weights_path: Path to pre-trained model.
       resnet_keras_weights: path to pretrained weights for the ResNet model of Keras (used only if version=keras).

    # Returns
       model: A Model instance.
    """

    # Create model
    input_shape = (img_height, img_width, img_channels)
    model = network(input_shape, n, num_classifiers, classifier_conf, version, num_classes=num_classes,
                    resnet_keras_weights=resnet_keras_weights)

    # Load the weights of the model
    model = load_weights(model, weights_path)
    return model


def train_model(opts, train_data_generator, val_data_generator, model, initial_epoch):

    """
    Model training.

    # Arguments
       train_data_generator: Training data generated batch by batch.
       val_data_generator: Validation data generated batch by batch.
       model: A Model instance.
       initial_epoch: Epoch from which training starts.
    """

    model.compile(loss=WeightedBinaryCrossEntropy(),  # (1, num_classifiers)  |  loss='binary_crossentropy',
                  optimizer=Adam(learning_rate=lr_schedule(0, opts.initial_lr)),
                  metrics=['binary_accuracy'])

    # Save model with the lowest validation loss
    weights_path = os.path.join(opts.save_dir, 'weights_{epoch:03d}.h5')
    write_best_model = ModelCheckpoint(filepath=weights_path, monitor='val_loss', save_best_only=True,
                                       save_weights_only=True)

    # Steps
    steps_per_epoch = int(np.ceil(train_data_generator.samples / opts.batch_size))
    validation_steps = int(np.ceil(val_data_generator.samples / opts.batch_size)) - 1

    # Other callbacks
    lr_scheduler = LearningRateScheduler(lr_schedule)
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
    tensorboard = TensorBoard(log_dir=os.path.join(opts.save_dir, 'log_dir'), write_graph=True, write_images=True,
                              update_freq='epoch', histogram_freq=0)
    hist_callback = HistoryCallback(os.path.join(opts.save_dir, 'log_dir'))
    callbacks = [write_best_model, lr_reducer, lr_scheduler, tensorboard, hist_callback]

    # Train the model
    try:
        model.fit(train_data_generator,
                  epochs=opts.epochs,
                  steps_per_epoch=steps_per_epoch,
                  callbacks=callbacks,
                  validation_data=val_data_generator,
                  validation_steps=validation_steps,
                  initial_epoch=initial_epoch,
                  use_multiprocessing=False,
                  # max_queue_size=32,
                  workers=16)
    except KeyboardInterrupt:
        model.save(os.path.join(opts.save_dir, 'weights_XXX.h5'))

    # Load history
    if os.path.isfile(os.path.join(opts.save_dir, 'log_dir', 'history.pkl')):
        with open(os.path.join(opts.save_dir, 'log_dir', 'history.pkl'), 'rb') as f:
            history = pickle.load(f)

        # Plot history
        elements = [e for e in list(history[0].keys()) if not e.startswith('val_')]
        for element in elements:
            hist_train, hist_val = [], []
            for k, v in history.items():
                hist_train.append(v[element])
                if 'val_' + element in list(history[0].keys()):
                    hist_val.append(v['val_' + element])
            plt.plot(hist_train)
            plt.plot(hist_val)
            plt.title('model ' + element)
            plt.ylabel(element)
            plt.xlabel('epoch')
            if len(hist_val) > 0:
                plt.legend(['train', 'val'], loc='upper left')
            else:
                plt.legend(['train'], loc='upper left')
            plt.show()
            plt.clf()
    print('Finished')

    
def main(opts):

    # Set training phase
    K.set_learning_phase(TRAIN_PHASE)

    # Create save_dir if not exists and restore_model == False
    time_txt = time.strftime("%Y%m%dT%H%M%S")
    if not opts.restore_model:
        aux = 6 if opts.rn_version == '1' else 9
        opts.save_dir = os.path.join(
            opts.save_dir,
            'ResNet{}{}c{}_{}_{}'.format(
                'Keras' if opts.rn_version == 'keras' else aux * opts.num_layers + 2,
                '' if opts.rn_version == 'keras' else 'v{}'.format(opts.rn_version),
                opts.classifier_conf,
                opts.dataset_name.replace('/', '_'),
                time_txt
            )
        )
        if not os.path.exists(opts.save_dir):
            os.makedirs(opts.save_dir)

    # Print and save options
    print('\nOptions:')
    if not os.path.exists(os.path.join(opts.save_dir, 'log_dir')):
        os.makedirs(os.path.join(opts.save_dir, 'log_dir'))
    save_opts = open(os.path.join(opts.save_dir, 'log_dir', 'options_{}.txt'.format(time_txt)), 'w')
    for k, v in vars(opts).items():
        save_opts.write("'{}': {}\n".format(k, v))
        print("'{}': {}".format(k, v))
    print()
    save_opts.close()

    # Input image dimensions
    img_width, img_height = opts.img_width, opts.img_height

    # Image mode (RGB or grayscale)
    if opts.img_mode == 'rgb':
        img_channels = 3
    elif opts.img_mode == 'grayscale':
        img_channels = 1
    else:
        raise IOError("Unidentified image mode: use 'grayscale' or 'rgb'")

    # Load file containing the path of the training images
    lines_train = load_lines(opts.train_imgs, opts.dataset_path)
    num_train = len(lines_train)
    if os.path.isfile(opts.val_imgs):
        lines_val = load_lines(opts.val_imgs, opts.dataset_path)
        num_val = len(lines_val)
    else:
        num_val = int(num_train * opts.val_perc)
        num_train -= num_val
        lines_val = lines_train[num_train:]
        lines_train = lines_train[:num_train]
    print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, opts.batch_size))

    # Reduce the number of classifiers
    filter_classifiers = opts.dataset_path + opts.filter_img if opts.filter_classifiers else None

    # Generate training data with real-time augmentation
    train_generator = data_utils.DataGenerator(lines_train,
                                               opts.num_classes,
                                               opts.v_grid,
                                               opts.h_grid,
                                               shuffle=True,
                                               img_mode=opts.img_mode,
                                               target_size=(img_height, img_width),
                                               batch_size=opts.batch_size,
                                               filter_img=filter_classifiers,
                                               data_aug=opts.data_aug)

    # Generate validation data with real-time augmentation
    val_generator = data_utils.DataGenerator(lines_val,
                                             opts.num_classes,
                                             opts.v_grid,
                                             opts.h_grid,
                                             img_mode=opts.img_mode,
                                             target_size=(img_height, img_width),
                                             batch_size=opts.batch_size,
                                             filter_img=filter_classifiers)

    # Number of classifiers based on grid size
    num_classifiers = train_generator.grid.shape[0]

    # Weights to restore and epoch from which training starts
    initial_epoch, weights_path = 0, os.path.join(opts.save_dir, opts.weights)
    if opts.restore_model:
        initial_epoch = opts.initial_epoch  # In this case weights are initialized as specified in pre-trained model
    else:
        weights_path = None  # In this case, weights are initialized randomly

    # Define model
    model = create_model(opts.num_layers, opts.rn_version, opts.classifier_conf, img_width, img_height, img_channels,
                         opts.num_classes, num_classifiers, weights_path, opts.resnet_keras_weights)

    # Serialize model into json
    model2json(model, os.path.join(opts.save_dir, opts.json_model))

    # Train model
    train_model(opts, train_generator, val_generator, model, initial_epoch)


if __name__ == "__main__":
    main(get_options())
