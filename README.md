# YOLOv3 with PEDL Client

A PEDL Client implementation of the [ultralytics/yolov3](https://github.com/ultralytics/yolov3) implementation in PyTorch.

## Setup

PEDL Client aims to replicate a local development experience and thus we start by creating a local copy of determined.ai's fork of the `ultralytics/yolov3` github repo. 

```
git clone --depth 1 git@github.com:determined-ai/yolov3.git
```

#### Data Download
To successfully use PEDL Client we need to make sure that all data is available both to your local environment and to the PEDL master. One way is to simply download the data to both your development local machine and to your PEDL master machine. Depending on your setup, you might be able to only download the data to one shared directory or simply point to a location that already has the data available. For simplicity, we just download the data to both machines by using the `bash get_coco2017.sh` script from within the `data` directory of this repo. 

Once downloaded, we need to edit `data/coco/train2017.txt` and `data/coco/val2017.txt` by replacing `../` by `data/` in front of each image file name. 

To make the data available to our script that runs with PEDL we use bind mounts in the experiment configuration:

```
config = {"bind_mounts": [{"host_path": "<abs_host_path_of_data>", "container_path": "data"}]}
```

#### Running it locally
<!-- Question: How does it know which libraries need to be installed remotely? -->
Now that we have the data available, we can run the script locally (outside of PEDL) to make sure everything works. We do this by using the following steps:

 1. Create Virtual Environment
 2. Install requirements with `pip install -U -r requirements.txt` and `pip install torchvision zmq`

To verify it works run the (original) train script via:
```
python train.py
```

### Running it with PEDL

We first install the PEDL command line interface. Then, we port over the original code to `pedl_train.py`. To port the code to PEDL, we need to implement a trial class, which we call `YOLOv3Trial` that inherits from PEDL's `PyTorchTrial` class. In this class, we define the model, the learning rate scheduler, the optimizer and the training and evaluation loop. 

#### Creating the Model
Overall, we use the same model architecture as the original `train.py` script. One temporary point of friction is that in PEDL the model is defined independently of the dataset and does not have access to the dataset directly. To circumvent this problem, we write a `LazyModule` class that shares data between different functions of the script. Going forward, there will be better ways of sharing data between model and data. 

#### Creating the Learning Rate Scheduler
The original `train.py` script mostly uses PyTorch's `MultiStepLR` scheduler. However, for the first three epochs it uses a higher warmup learning rate. To mimic this behavior, we create our own learning rate scheduler `WarmupMultiStepLR` which can set a higher learning rate for the first few epochs.

### Training and Evaluating

We feed the training and evaluation data into PEDL by replacing the original `torch.utils.data.DataLoader` by PEDL's `DataLoader`. Apart from that, no code changes are necessary to load the data into PEDL. We also define a `train_batch` function which replicates the computation of one training step in the original `train.py` file. Again, no further code changes are necessary. Similarly, we create a `evaluate_full_dataset` function that computes test metrics on the entire dataset. 

### Configs and Parameters
Finally, we make sure that all parameters of the original model are also available to pedl. By default, all variables set in the `__main__` part of `pedl_train.py` are available to all function and we can just use them as we would during local development. To take advantage of PEDL hyperparameter tuning, we can expose some of the hyperparameters to PEDL by defining them as `pedl.Constant`s and then updating the previously defined parameters. 

For example, the original model had a `batch_size` hyperparameter which was part of the `opt` namespace. To turn this hyperparameter into a PEDL hyperparameter, we first set the batch_size as a PEDL constant and then overwrite the namespace variable:

```
batch_size = pedl.Constant(value=16, name="batch_size")
opt.batch_size = batch_size
```
Lastly, we set the searcher metric to use `mAP` and set the number of steps such that it equals 273 epochs. Once this is set up we start the experiment by running the `pedl_train.py` script via:

```
python pedl_train.py
```

This triggers the experiment creation from within the file:
```
exp = pedl.Experiment(
    context_directory=str(pathlib.Path.cwd()),
    trial_def=YOLOv3Trial,
    make_data_loaders_fn=make_data_loaders,
    configuration=config,
)
exp.create()
```
Further implementation details are available in the [`pedl_train.py`](pedl_train.py) script.