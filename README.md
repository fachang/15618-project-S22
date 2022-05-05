# 15618-Project-S22

## Summary

Explore how to design the parallelism for deep learning modules on different devices and their toolkits. We focus on comparing and profiling METAL SHADING version on the iOS devices (iPhone, iPad) with CUDA version on GHC machines.

## Background

With the great success of deep learning in different applications, there is an increasing trend to deploy deep learning model on different devices. For example, if a deep learning model is deployed on a mobile phone, a user can use the phone camera to identity objects directly without Internet connection. However, the **limited resources** of mobile devices make it challenging to parallelize the deep network prediction. Also, as the network gets deeper, which is the trend in deep learning, is the parallelism design still scalable?

In this project, we plan to implement both **METAL SHADING** and **CUDA** parallel version of **network forwarding** for modules related to image classification, and then conduct a **profiling and review** on the implementations and devices. Through this analysis, we hope to identify the **scalability or limitation** of mobile GPU hardwares compared to CUDA hardwares and their **parallel design differences**.

## Challenge
It is undeniable that the technique of machine learning especially deep learning solve the real problems efficiently. However, despite of the fact that the models' accuracy are very well, some cannot really deployed on the platform like iPhone or iPad because of the fact that the computation of deep learning is too expensive. We want to explore the method of how to parallelize the computation well enough to make the model can be run on the platform in real-time to enhance the user experience. For examplem, what kind of parallelized method may be more suitable for Metal or CUDA.
## Resources
GHC machine, iPhone, iPad and MacBook

## Goals and deliverables
For 75% targets, we want to implement all required computation units like convolution unit in Metal and CUDA to profile the performance versus the parallelized method. 
For 100% targets, we want to implemente the well known object classification model using the layers we created to see the inference performance.
For 125% targets, we want to implemente the well known object classification model using the layers we created to see the training performance.

## Schedule

| Week     | Task    | Status |Milestone|
| -------- | -------- | ------- |---------|
| 3/20     | ~~Sequential version CNN units implementation in metal~~ Implement image classification model using layers we created     |     |3/23 Proposal|
| 3/27     | ~~Parallel version CNN units implementation in metal~~ Parallel version CNN units implementation in CUDA    |
| 4/3     | Unit testing and refactor||4/11 Checkpoint|
| 4/10     | ~~Implement image classification model using layers we created~~ Sequential version CNN units implementation in metal           ||
| 4/17     | ~~Parallel version CNN units implementation in CUDA~~ Parallel version CNN units implementation in metal         ||
| 4/24     | Compare implementation of CUDA, METAL and profiling, Report ||4/29 Report|
| 5/01     | Prepare for presentation ||5/05 Presentation|

### Tasks
1. Model(Weight initialization)
2. Sequential CNN(Convolution, linear, pooling, relu, softmax) metal shading + metal
3. Parallel CNN metal shading
4. Parallel CNN CUDA
5. Compare implementation of CUDA, METAL

## 04/11 Milestone

### The work that we have completed so far
We implemented the sequential version and basic cuda version of convolution, ReLU, pooling and fully-connected layers. We feed the same input, weight and bias and get the same output value as pytorch gives us.


We also implemented a benchmark iOS App for demostrating the performance metrics and a CPU version linear layer by Swift.

### Current progress w.r.t. the goals in proposal
Due to the fact that we find it is a little bit hard to get enough information about Metal and how to setup XCode environment, we decided to implement the function in CUDA first instead of Metal first. Now, we are trying to setup XCode environment to allow us to implement the function in Swift and Metal.

### Updated list of goals for the poster session
The goala are not going to be changed so far.
### What to show at Poster session 
We would like to compare the speedup with Metal and without Metal on iOS device. Furthermore, we also want to compare the parallel method when it comes to implementing in CUDA and in Metal.
#### Preliminary result at this time
We have implemented CUDA version for CNN and the following is the speedup using the first layer of LeNet. We make sure the output value is excatly the same as the one generated by pytorch.
![](https://i.imgur.com/6rBL5k3.png)

### Issues that concern us the most
The Metal implementation concern us the most because we find that its documentation resource is rarely limited. Although for now, we have read its official document, but we are not sure whether we will encounter problems which not stated in the document in the future.

### Final report data
![image](https://user-images.githubusercontent.com/7065983/166180549-c78abdfd-5435-4405-aeec-6ca7d817f96b.png)


![image](https://user-images.githubusercontent.com/7065983/166180563-15688e2a-3cc0-419e-9dee-0a72286de3b6.png)

![image](https://user-images.githubusercontent.com/7065983/166180621-a8847f09-6cc3-4bc6-9fa8-025b20b99474.png)

### Link for pretrained model
https://drive.google.com/drive/folders/1ynqzZki838f2HIABVC8DEBW6HvlWuw6_?usp=sharing

## 05/04 Final Stage
### Project Video
https://drive.google.com/file/d/1U7zKuFRD_NlG1MIacHqNULXSsJGvw47W/view?usp=sharing



