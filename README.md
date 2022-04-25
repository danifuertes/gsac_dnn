**GSAC-DNN**

Punctual detections are a good alternative to bounding box detections because they speed up the deployment of datasets
(only one point is required per instance). This project can be used to automatically detect instances from images. Our
neural network, [GSAC-DNN](https://www.sciencedirect.com/science/article/pii/S1051200422000902?via%3Dihub)
(Grid of Spatial Aware Classifiers - Deep Neural Network), is an adaptation of the previous work
[GSAC](https://www.sciencedirect.com/science/article/abs/pii/S0923596521000023). GSAC-DNN network contains a backbone
based on ResNet and a grid of classifiers based on fully connected layers. During the training, ResNet extracts feature
maps from the input images and the grid of classifiers learns to focus their attention on their correspondent location of
the feature maps. At test time, we added NMS (Non-Maxima Suppression) to remove repeated predictions, and weighted
interpolation to extract the exact position of the marker from a neighborhood of classifiers whose confidence score
surpasses a given threshold.

If this repository is useful for your work, please cite our paper:

```
@article{FUERTES2022103473,
title = {People detection with omnidirectional cameras using a spatial grid of deep learning foveatic classifiers},
journal = {Digital Signal Processing},
volume = {126},
pages = {103473},
year = {2022},
issn = {1051-2004},
doi = {https://doi.org/10.1016/j.dsp.2022.103473},
url = {https://www.sciencedirect.com/science/article/pii/S1051200422000902},
author = {Daniel Fuertes and Carlos R. del-Blanco and Pablo Carballeira and Fernando Jaureguizar and Narciso García}
}
``` 

**Software requeriments**

This code has been tested on Ubuntu 18.04.6 LTS with Docker 20.10.12, Python 3.6.9, TensorFlow 2.4.0, CUDA 11.0 and a
GPU TITAN Xp. The dependencies can be obtained as follows:

1. Build the Docker image with `docker build -t name_image .`
2. Run the Docker container with `docker run --user $(id -u):$(id -g) --gpus all -it --rm --volume=$(pwd):/home/inpercept:rw --volume=/path/to/dataset:/path/to/dataset:ro --name name_container name_image bash`

This code can also be executed without docker:

1. Install Python 3.6.9. Other versions may also be valid.
2. Create a virtual environment with `virtualenv --python=/path/to/python/interpreter venv`. Usually, the python
interpreter is located in `/usr/bin/python3.6.9`.
3. Activate the virtual environment with `source venv/bin/activate`.
4. Install the dependencies with `pip3 install -r requirements.txt`.
5. Install TensorFlow 2.4.0 and CUDA 11.0 following [these instruccions](https://www.tensorflow.org/install/gpu)

**Dataset**

Your dataset should be divided on 2 main sets: train and test. The structure of your dataset should be similar to the
following one:
```
/DatasetName
    train.txt
    test.txt
    /train
        /train_sequence_1
            000000000.png
            000000001.png
            000000002.png
            ...
        /train_sequence_2
          ...
        /train_sequence_N
          ...
    /test
        /test_sequence_1
            000000000.png
            000000001.png
            000000002.png
            ...
        /test_sequence_2
          ...
        /test_sequence_N
          ...
```
The `train.txt` and `test.txt` files should contain a list of the train and test annotations, respectively. The
ground-truth format is described next:
```
/relative/path/to/train_sequence_1/000000000.png x,y
/relative/path/to/train_sequence_1/000000001.png x,y x,y x,y
/relative/path/to/train_sequence_1/000000002.png
/relative/path/to/train_sequence_1/000000000.png x,y x,y
...
/relative/path/to/train_sequence_2/000000000.png x,y
/relative/path/to/train_sequence_2/000000001.png
/relative/path/to/train_sequence_2/000000002.png x,y x,y x,y x,y
...
```
where x and y are the coordinates of each of the point-based annotations on the image. You can configure your validation
data like your train and test data. The file with the annotations of your validation set should be called val.txt. If
your dataset does not contain a validation set, you can provide a percentage with the option `--val_perc` to extract
some random samples from the training set and use them to validate:

```bash
python train.py --dataset_path /path/to/directory/containing/your/dataset --dataset_name DatasetName --val_perc 0.1 --img_width 224 --img_height 224 --h_grid 28 --v_grid 28
```

In case you have a validation set with a format similar to the one described above, you can train your model with:

```bash
python train.py --dataset_path /path/to/directory/containing/your/dataset --dataset_name DatasetName --img_width 224 --img_height 224 --h_grid 28 --v_grid 28
```

While you are training a model, the weights that optimize the validation loss are saved in 
`models/model_DatasetName_CurrentDate` by default, where `model` is `ResNet{}v{}c{}` (`{}` corresponds to the number of
layers, the ResNet version and the type of classifiers, respectively). To restore a model, you should use the option
`--restore_model True`, indicate the path to the model with `--save_dir models/model_DatasetName_TrainDate` and indicate
the weights desired with `--weights weights_057.h5`. Example:

```bash
python train.py --restore_model True --save_dir models/model_DatasetName_TrainDate --weights weights_057.h5 --dataset_path /path/to/directory/containing/your/dataset --dataset_name DatasetName --img_width 224 --img_height 224 --h_grid 28 --v_grid 28
```

For any additional help, you can run:

```bash
python train.py --help
```

**Test**

To evaluate your trained model using your test data with the format described above, you can run:

```bash
python test.py --save_dir models/model_DatasetName_TrainDate --weights weights_057.h5 --dataset_path /path/to/directory/containing/your/dataset --dataset_name DatasetName --img_width 224 --img_height 224 --h_grid 28 --v_grid 28
```

Note that options related to the structure of the network should not be changed. In case you do not remember any of the
options, read the file `models/model_DatasetName_TrainDate/log_dir/options.txt`, that contains a list with the options
used to train that model.

To visualize the detections, you can use the option `--visualize True`. Additionally, you can see also the predictions
of the grid of classifiers using `--view_grid True`. Example:

```bash
python test.py --visualize True --view_grid True --save_dir models/model_DatasetName_TrainDate --weights weights_057.h5 --dataset_path /path/to/directory/containing/your/dataset --dataset_name DatasetName --img_width 224 --img_height 224 --h_grid 28 --v_grid 28
```

You can save the images with the predictions by running:

```bash
python test.py --save_imgs True --save_dir models/model_DatasetName_TrainDate --weights weights_057.h5 --dataset_path /path/to/directory/containing/your/dataset --dataset_name DatasetName --img_width 224 --img_height 224 --h_grid 28 --v_grid 28
```

The images are saved by sequences in `map/predictions/model_DatasetName_TrainDate_test_TestDate/images`. Next to this
directory, you can find 2 directories `called detection-results` and `ground-truth`. These directories contain files
with the predictions and annotations of each test image, respectively. To evaluate your model with metrics like Precision
(P), Recall (R), F1-Score (F), mean Average Precision (mAP), Miss Rate (MR), False Positives Per Images (FPPI), and Log
Average Miss Rate (LAMR), it is necessary to run another script:

```bash
python -m map.map --results_dir map/predictions/model_DatasetName_TrainDate_test_TestDate --img_width 224 --img_height 224
```

Check the folder `map/predictions/model_DatasetName_TrainDate_test_TestDate/results` to find the results computed.

**Note**

At the beginning of `train.py` and `test.py` you can set the device changing the value of
`os.environ["CUDA_VISIBLE_DEVICES"]`: CPU="-1", first GPU="0", second GPU="1", etc.
