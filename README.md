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

* Distributed Training:
  * PyTorch distributed framework:
    * [Data parallel training with PyTorchTrainingJob](./job/pytorchtrainingjob/ddp/)
    * [Parameter server training with PyTorchTrainingJob](./job/pytorchtrainingjob/ps/)
  * DeepSpeed:
    * [Data parallel training with DeepSpeedJob](./job/deepspeedjob/)
  * ColossalAI:
    * [Data parallel training with ColossalAIJob](./job/colossalaijob/)
  * TensorFlow distributed framework:
    * [Data parallel training with TensorFlowTrainingJob](./job/tensorflowtrainingjob/multiworker/)
    * [Parameter server training with TensorFlowTrainingJob](./job/tensorflowtrainingjob/ps/)
  * Horovod:
    * [Horovod data parallel training with MPIJob (Keras model)](./job/mpijob/horovod-keras/)
    * [Horovod data parallel training with MPIJob (PyTorch model)](./job/mpijob/horovod-torch/)
  * XGBoost:
    * [Distributed training and prediction with XGBoostTrainingJob](./job/xgboosttrainingjob/distributed/)
  * Apache Beam:
    * [Run Apache Beam distributed computing tasks with BeamJob](./job/beamjob/count-word/)
  * Custom distributed training:
    * [Data parallel training with GenericJob (Keras model)](./job/genericjob/multiworker-keras/)
  * [Debug Job](./job/debug/)
* Automatic Hyperparameter Tuning:
  * [HPO with AutoTune (Keras model)](./autotune/hpo-keras/)
  * [HPO with AutoTune (PyTorch model)](./autotune/hpo-torch/)
* Record Training Data and Metadata:
  * [Record training with EM for single-worker training](./em/single-worker-training/)
  * [Record training with EM for data parallel training](./em/data-parallel-training/)
* Model Deployment:
  * MLService:
    * [Deploy PyTorch model from PVC](./deployment/mlservice/torch-pvc/)
    * [Deploy TensorFlow model with Transformer](./deployment/mlservice/transformer/)
  * SimpleMLService:
    * [Deploy TensorFlow model](./deployment/simplemlservice/)
* Workflow:
  * [Build end-to-end workflow from data sampling to model export](./workflow/e2e-workflow/)
* Platform Tools:
  * [Codepack usage example](./codepack/)
* Other Features:
  * [Build image on platform](./build-image/build-image-on-platform/)
  * [Build Notebook custom image](./build-image/build-notebook-custom-image/)
