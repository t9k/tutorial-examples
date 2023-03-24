# Tutorials Examles

<div id="top" align="center">

<img src="./assets/illustration.png" alt="illustration" width="300" align="center"><br>

| [English](README.md) | [中文](docs/README-zh.md) |

</div>

This git repository stores all tutorial examples of **TensorStack AI Computing Platform**. The majority of examples are based on the "*MNIST Handwritten Digits recognition task*", which is a familiar and convenient task for users to begin exploring the platform's various functions.

## Usage

1. In your project, create a PVC named `tutorial` with a size of 4 Gi or more, and then create a Notebook also named `tutorial` to mount the PVC, and the image and resources are not limited (For the steps to create a PVC and Notebook, please refer to [Create a Notebook](https://t9k.github.io/user-docs/guide/develop-and-test-model/create-notebook.html); if you want to access your notebook remotely, please toggle SSH).

2. Enter Notebook or remotely connect to Notebook, start a terminal, execute the following command to clone this repository (see [Using Notebook](https://t9k.github.io/user-docs/guide/develop-and-test-model/use-notebook.html)).

```shell
cd ~
git clone https://github.com/t9k/tutorial-examples.git
```

3. Continue with **Notebook terminal** and refer to the README of each example, or refer to the corresponding tutorial in the user documentation of each example. The README of each example is consistent with the operation steps given in the corresponding tutorial.

## Examples

* Distributed Training
  * TensorFlow
    * [TensorFlowTrainingJob parallel training with multi-worker](./job/tensorflowtrainingjob/multiworker/)
    * [TensorFlowTrainingJob parallel training with parameter server](./job/tensorflowtrainingjob/ps/)
  * PyTorch
    * [PyTorchTrainingJob parallel training with multi-worker](./job/pytorchtrainingjob/ddp/)
    * [PyTorchTrainingJob parallel training with parameter server](./job/pytorchtrainingjob/ps/)
  * Horovod
    * [Horovod - Keras parallel training with multi-worker](./job/mpijob/horovod-keras/)
    * [Horovod - PyTorch parallel training with multi-worker](./job/mpijob/horovod-torch/)
  * XGBoost
    * [XGBoostTrainingJob parallel training](./job/xgboosttrainingjob/distributed/)
  * Apache Beam
    * [BeamJob for parallel data procesing with Apache Beam](./job/beamjob/count-word/)
  * Custom
    * [GenericJob - Keras parallel training with multi-worker](./job/genericjob/keras-multiworker/)
* HyperParameter Tuning
  * [AutoTune - HPO for Keras](./autotune/hpo-keras/)
  * [AutoTune - HPO for PyTorch）](./autotune/hpo-torch/)
* Recording Training metadata
  * [Using AIMD with a single worker](./aimd/single-worker/)
  * [Using AIMD with Job](./aimd/job/)
* Model Depolyment
  * Models stored in PVC
    * [Keras](./deployment/pvc/mlservice-keras/)
    * [PyTorch](./deployment/pvc/mlservice-torch/)
  * Models stored in S3
    * [Keras，S3](./deployment/s3/mlservice-keras/)
    * [PyTorch, S3](./deployment/s3/mlservice-torch/)
* Workflow
  * [E2E Workflow - from processing data to deploying model](./workflow/automatic-workflow/)
* Tools
  * [Codepack ](./codepack/)
* Others
  * [Build container images](./build-custom-image/)
