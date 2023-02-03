# [IROS 2022] VPEC: Visual Pressure Estimation and Control for Soft Robotic Grippers

## Abstract
Soft robotic grippers facilitate contact-rich manipulation, including robust grasping of varied objects. Yet the beneficial compliance of a soft gripper also results in significant deformation that can make precision manipulation challenging. We present visual pressure estimation & control (VPEC), a method that infers pressure applied by a soft gripper using an RGB image from an external camera. We provide results for visual pressure inference when a pneumatic gripper and a tendon-actuated gripper make contact with a flat surface. We also show that VPEC enables precision manipulation via closed-loop control of inferred pressure images. In our evaluation, a mobile manipulator (Stretch RE1 from Hello Robot) uses visual servoing to make contact at a desired pressure; follow a spatial pressure trajectory; and grasp small low-profile objects, including a microSD card, a penny, and a pill. Overall, our results show that visual estimates of applied pressure can enable a soft gripper to perform precision manipulation.

## Project Description

[[Paper]](https://arxiv.org/abs/2204.07268) [[5-minute IROS Video]](https://www.youtube.com/watch?v=Hq6GA6W5QRI)

This is the repository for VPEC, a project to estimate the pressure between a soft gripper and a flat surface using ONLY a single image. We use a neural network to input the single RGB frame, and output a pressure map showing where pressure occurs in image space. We hypothesize that the network attends to small changes in the image, such as shadows and gripper deformation, to estimate pressure.

Our network (VPEC-Net) takes a single RGB frame (top), and estimates image-space pressure (bottom right). Ideally, the estimated pressure matches the ground-truth pressure as measured by a sensor (bottom left).

![Animation of Pressure Estimation](docs/pressure_animation.gif)

We then use these pressure estimates to grasp small objects. The robot uses the pressure estimates to regulate force into the table and grasp the object. Even though the gripper bends significantly during these trials, our pressure estimates allow the robot to visually servo and reliably grasp the small objects. 

![Animation of Autonomous Grasping](docs/grasping_animation.gif)


## Installing Code and Data

VPEC requires Python and PyTorch, and has been tested with Python 3.10 and Pytorch 1.12 on Ubuntu. The following commands can be used to create a fresh conda environment and install all dependencies, however installing PyTorch through any other way is acceptable:
```
conda create -n vpec python=3.10
conda activate vpec
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
```

Next, download the collected dataset and model files using the following script. The total dataset is about 170 GB, and will be placed in the `data` directory:
```
python -m recording.downloader
``` 

## Getting Started

To run VPEC-Net on the test set of the dataset and generate a video of the results and save it in the `data/movies` folder, run the following command. Replace `stretch` with `soft` depending on which gripper you want to evaluate:

```
python -m prediction.make_network_movie --config stretch_nopinch
``` 

To calculate the numbers reported in the paper, run:
```
python -m prediction.evaluator --config stretch_nopinch
```

## Training a model

To train a model, run the following command
```
python -m prediction.trainer --config stretch_nopinch
```