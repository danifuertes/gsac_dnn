"""ResNet models: version 1 and 2
ResNet v1
[a] Deep Residual Learning for Image Recognition
https://arxiv.org/pdf/1512.03385.pdf
ResNet v2
[b] Identity Mappings in Deep Residual Networks
https://arxiv.org/pdf/1603.05027.pdf
"""

from __future__ import print_function
from tensorflow import keras
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Activation, AveragePooling2D, Input, Flatten,\
    Dropout, Concatenate
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
# from keras.utils.vis_utils import plot_model
from tqdm import tqdm


def lr_schedule(epoch, init_lr=1e-3):

    """Learning Rate Schedule
    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.
    # Arguments
        epoch (int): The number of epochs
        init_lr (float32): initial learning rate
    # Returns
        lr (float32): learning rate
    """

    lr = init_lr
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr


def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):

    """
    2D Convolution-Batch Normalization-Activation stack builder.
    # Arguments
        inputs (tensor): input tensor from input image or previous layer.
        num_filters (int): Conv2D number of filters.
        kernel_size (int): Conv2D square kernel dimensions.
        strides (int): Conv2D square stride dimensions.
        activation (string): activation name.
        batch_normalization (bool): whether to include batch normalization.
        conv_first (bool): conv-bn-activation (True) or activation-bn-conv (False).
    # Returns
        x (tensor): tensor as input to the next layer
    """

    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x


def resnet_keras(inputs, input_shape, weights, num_classes=1):
    """
    ResNet Version from Keras
    # Arguments
        inputs (tensor): input image tensor.
        input_shape (tensor): shape of input image tensor.
        weights (str): path to pretrained weights.
                       If weights == 'imagenet' => pretrained weights on ImageNet are downloaded.
        num_classes (int): number of output classes.
    # Returns
        model (Model): Keras model instance
    """

    x = keras.applications.resnet50.ResNet50(
        include_top=False, weights=weights, input_tensor=inputs,  # weights='imagenet'
        input_shape=input_shape, pooling=None, classes=num_classes
    )

    return x.output


def resnet_v1(inputs, depth):

    """
    ResNet Version 1 Model builder [a]
    Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
    Last ReLU is after the shortcut connection.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filters is
    doubled. Within each stage, the layers have the same number filters and the
    same number of filters.
    Feature maps sizes:
    stage 0: 32x32, 16
    stage 1: 16x16, 32
    stage 2:  8x8,  64
    The Number of parameters is approx the same as Table 6 of [a]:
    ResNet20 0.27M
    ResNet32 0.46M
    ResNet44 0.66M
    ResNet56 0.85M
    ResNet110 1.7M
    # Arguments
        input_shape (tensor): input image tensor.
        depth (int): number of core convolutional layers.
    # Returns
        model (Model): Keras model instance
    """

    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    # Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    x = resnet_layer(inputs=inputs)

    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):

            strides = 1

            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample

            y = resnet_layer(inputs=x,
                             num_filters=num_filters,
                             strides=strides)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters,
                             activation=None)

            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)

            x = keras.layers.add([x, y])
            x = Activation('relu')(x)

        num_filters *= 2

    # v1 does not use BN after last shortcut connection-ReLU
    x = AveragePooling2D(pool_size=8)(x)
    return x


def resnet_v2(inputs, depth):

    """
    ResNet Version 2 Model builder [b]
    Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D or also known as
    bottleneck layer
    First shortcut connection per layer is 1 x 1 Conv2D.
    Second and onwards shortcut connection is identity.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filter maps is
    doubled. Within each stage, the layers have the same number filters and the
    same filter map sizes.
    Features maps sizes:
    conv1  : 32x32,  16
    stage 0: 32x32,  64
    stage 1: 16x16, 128
    stage 2:  8x8,  256
    # Arguments
        inputs (tensor): input image tensor.
        depth (int): number of core convolutional layers.
    # Returns
        model (Model): Keras model instance
    """

    if (depth - 2) % 9 != 0:
        raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
    # Start model definition.
    num_filters_in = 16
    num_res_blocks = int((depth - 2) / 9)

    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    x = resnet_layer(inputs=inputs,
                     num_filters=num_filters_in,
                     conv_first=True)

    # Instantiate the stack of residual units
    for stage in range(3):
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:  # first layer and first stage
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                if res_block == 0:  # first layer but not first stage
                    strides = 2    # downsample

            # bottleneck residual unit
            y = resnet_layer(inputs=x,
                             num_filters=num_filters_in,
                             kernel_size=1,
                             strides=strides,
                             activation=activation,
                             batch_normalization=batch_normalization,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_in,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_out,
                             kernel_size=1,
                             conv_first=False)
            if res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters_out,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = keras.layers.add([x, y])

        num_filters_in = num_filters_out

    # v2 has BN-ReLU before Pooling
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=8)(x)
    return x


def classifiers(x, num_classifiers, design, num_classes=1):
    """
    Choose and implement a configuration for the structure of the classifiers.

    If design == 1 each classifiers consists of:
        Dense layer + Dropout
        Dense layer + Dropout
        Dense layer

    If design == 2:
        ResNet layer
        Dense layer + Dropout
        Dense layer

    If design == 3:
        ResNet layer
        Dense layer

    If design == 4:
        Dense layer

    # Arguments
        x (tensor): feature maps from the backbone.
        num_classifiers (int): number of classifiers that compose the grid.
        design (int): ID to identify the architecture of the classifiers.
        num_classes (int): number of output classes.

    # Returns
        outputs (list): list containing the classifiers.
    """

    # List where each classifier is appended in order to deliver them to the output of the model
    outputs = []
    print('Loading classifiers...')

    if design == 1:

        x = Flatten()(x)
        for i in tqdm(range(num_classifiers)):

            # Create the outputs of the neural network (Classifiers)
            x_net = Dense(16, activation='relu')(x)
            x_net = Dropout(rate=0.5)(x_net)
            x_net = Dense(16, activation='relu')(x_net)
            x_net = Dropout(rate=0.5)(x_net)
            x_net = Dense(num_classes, activation='sigmoid', name='Classifier_' + str(i))(x_net)

            # Add the classifier to the outputs list
            outputs.append(x_net)

    elif design == 2:

        for i in tqdm(range(num_classifiers)):

            # Create the outputs of the neural network (Classifiers)
            x_net = resnet_layer(x)
            x_net = Flatten()(x_net)
            x_net = Dense(16, activation='relu')(x_net)
            x_net = Dropout(rate=0.5)(x_net)
            x_net = Dense(num_classes, activation='sigmoid', name='Classifier_' + str(i))(x_net)

            # Add the classifier to the outputs list
            outputs.append(x_net)

    elif design == 3:

        for i in tqdm(range(num_classifiers)):

            # Create the outputs of the neural network (Classifiers)
            x_net = resnet_layer(x)
            x_net = Flatten()(x_net)
            x_net = Dense(num_classes, activation='sigmoid', name='Classifier_' + str(i))(x_net)

            # Add the classifier to the outputs list
            outputs.append(x_net)

    else:

        x = Flatten()(x)
        for i in tqdm(range(num_classifiers)):

            # Create the outputs of the neural network (Classifiers)
            x_net = Dense(num_classes, activation='sigmoid', name='Classifier_' + str(i))(x)

            # Add the classifier to the outputs list
            outputs.append(x_net)

    return outputs


def network(input_shape, n, num_classifiers, classifier_conf, resnet_version, resnet_keras_weights=None,
            num_classes=1):

    # Check version parameters are correct
    assert resnet_version in ['1', '2', 'keras'], 'ResNet architecture version must be 1, 2 or keras'
    assert classifier_conf in [1, 2, 3, 4], 'Classifier architecture version must be an integer number between 1 and 4'

    # Define input tensor
    inputs = Input(shape=input_shape)

    # Choose ResNet version (Backbone)
    if resnet_version == '1':
        depth = n * 6 + 2  # Computed depth from supplied model parameter n
        feature_maps = resnet_v1(inputs, depth)
    elif resnet_version == '2':
        depth = n * 9 + 2  # Computed depth from supplied model parameter n
        feature_maps = resnet_v2(inputs, depth)
    else:
        depth = 0
        feature_maps = resnet_keras(inputs, input_shape, resnet_keras_weights, num_classes)

    # Model name, depth and version
    model_type = 'ResNet%dv%s' % (depth, resnet_version) if depth > 0 else 'ResNet50_keras_version'
    print(model_type)

    # Grid of Spatial Aware Classifiers (GSAC)
    outputs = classifiers(feature_maps, num_classifiers, classifier_conf, num_classes)
    outputs = Concatenate(name='out')(outputs)
    # o = []
    # for i in range(num_classifiers):
    #     o.append(outputs[..., i])

    # Define model
    model = Model(inputs=[inputs], outputs=outputs)
    print(model.summary())
    # plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    return model
