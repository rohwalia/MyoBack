# MyoBack model: Simulation of a neuromusculoskeletal model of the back with an exoskeleton

[Rohan Walia]()<sup>1</sup>,
[Morgane Billot]()<sup>1</sup>,
[Kevin Garzon-Aguirre]()<sup>1</sup>,
[Swathika Subramanian]()<sup>2</sup>,
[Huiyi Wang]()<sup>1</sup>,
[Mohamed Irfan Refai]()<sup>2</sup>,
[Guillaume Durandau]()<sup>1</sup>

<sup>1</sup>McGill University
<sup>2</sup>University of Twente

<img src="MyoBack_repo.png" alt="MyoBack" width="30%"/>

## Overview
MyoBack is a human back model part of the MyoSuite framework, derived from a physiologically accurate OpenSim model. MyoBack was validated empirically by integrating a passive back exoskeleton in simulation and comparing forces exerted on the back with values from experimental trials.

MyoBack model can be applied to a range of contexts, including both passive and active exoskeletons, or even to entirely different use cases. To give a basis, we have added an example where a RL agent is trained to control the muscles in the model to balance it and keep it from leaning forward.

The folder [rl_example](./rl_example) includes a sample training script and baseline with which the MyoBack model developed can be tested. Use the instructions below to set up and run the training scripts or load the baseline on your own machine.

To simply view the model, you can use the MuJoCo `simulate` GUI or the MuJoCo Python bindings with an example given in [test_dynamic.py](./test_dynamic.py).

## Installation
Install our conda environment on a Linux machine. On Ubuntu 20.04 you need to install the following apt packages for mujoco:
```console
$ sudo apt install -y libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf
```
We recommend [Mambaforge](https://github.com/conda-forge/miniforge#mambaforge) instead of the standard anaconda distribution for faster installation: 
```console
$ mamba env create -f conda_environment.yaml
```
but you can use conda as well: 
```console
$ conda env create -f conda_environment.yaml
```
## RL example
### Training the Agent
To start training the agent, use the following command:
```
cd rl_example
python train_back_RL.py --group myoback_1 --num_envs 1 --learning_rate 0.0002 --clip_range 0.1 --seed 7
```
- `--group`: Wandb training group name
- `--num_envs`: Number of envs to train in parallel
- `--learning_rate`: learning rate for PPO
- `--clip_range`: clip range for PPO
- `--seed`: env seeding


### Baseline Model Video
To view the performance of the baseline model, navigate to:
[baseline_side_video.mp4](./rl_example/videos/myoStandingBack-v0/baseline_side_video.mp4)

### Contact Information
For more information or assistance, contact:  <br />

Email: huiyi.wang@mail.mcgill.ca <br />
GitHub: cherylwang20
