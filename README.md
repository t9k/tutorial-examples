# Tutorial Examples

<div id="top" align="center">

<img src="./assets/illustration.png" alt="illustration" width="300" align="center"><br>

| [English](README.md) | [中文](docs/README-zh.md) |

</div>

This git repository stores all tutorial examples of **TensorStack AI Computing Platform**. The majority of examples are based on the "*MNIST Handwritten Digits recognition task*", which is a familiar and convenient task for users to begin exploring the platform's various functions.

## Usage

1. In your project, create a PVC named `tutorial` with a size of 4 GiB or more, and a Notebook with the same name. The new Notebook need to be configured to use the just-created PVC (mounted to `/t9k/mnt`). If you want to access your notebook remotely, please toggle SSH on. Other options for the notebook are at your choice. For more details on how to create a PVC and Notebook, please refer to [User doc: Create a Notebook](https://t9k.github.io/user-docs/guide/develop-and-test-model/create-notebook.html).

2. Open Notebook from Web UI or connect remotely using ssh to get a terminal. Then execute the following command to clone this repository (see [User doc: Use Notebook](https://t9k.github.io/user-docs/guide/develop-and-test-model/use-notebook.html)).

```
# change dir to $HOME, /t9k/mnt
cd
git clone https://github.com/t9k/tutorial-examples.git
```

3. Continue with your **Notebook terminal** and refer to the README of each example, or refer to the corresponding tutorial in the user documentation of each example. The README of each example is consistent with the steps given in the corresponding tutorial.

## Examples

* Distributed Training
  * TensorFlow
    * [Distributed data parallel training](./job/tensorflowtrainingjob/multiworker/)
    * [Distributed parameter server training](./job/tensorflowtrainingjob/ps/)
  * PyTorch
    * [Distributed data parallel training](./job/pytorchtrainingjob/ddp/)
    * [Distributed parameter server training](./job/pytorchtrainingjob/ps/)
  * Horovod
    * [Keras distributed data parallel training](./job/mpijob/horovod-keras/)
    * [PyTorch distributed data parallel training](./job/mpijob/horovod-torch/)
  * XGBoost
    * [XGBoostTrainingJob distributed training](./job/xgboosttrainingjob/distributed/)
  * Apache Beam
    * [BeamJob for distributed data processing with Apache Beam](./job/beamjob/count-word/)
  * Custom
    * [GenericJob - Keras distributed data parallel training](./job/genericjob/keras-multiworker/)
  * [Debugging with debug mode of Job](./job/debug-mode/)
* HyperParameter Tuning
  * [AutoTune - HPO for Keras](./autotune/hpo-keras/)
  * [AutoTune - HPO for PyTorch](./autotune/hpo-torch/)
* Recording Training metadata
  * [Using AIMD with a single worker](./aimd/single-worker/)
  * [Using AIMD with Job](./aimd/job/)
* Model Deployment
  * Models stored in PVC
    * [Keras](./deployment/pvc/mlservice-keras/)
    * [PyTorch](./deployment/pvc/mlservice-torch/)
  * Models stored in S3
    * [Keras，S3](./deployment/s3/mlservice-keras/)
    * [PyTorch, S3](./deployment/s3/mlservice-torch/)
* Workflow
  * [E2E Workflow - from processing data to deploying model](./workflow/automatic-workflow/)
* Tools
  * [Codepack](./codepack/)
* Others
  * [Build image on platform](./build-image/build-image-on-platform/)
  * [Build Notebook custom image](./build-image/build-notebook-custom-image/)
