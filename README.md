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
| 3/20     | Sequential version CNN units implementation in metal     |     |3/23 Proposal|
| 3/27     | Parallel version CNN units implementation in metal     |
| 4/3     | Unit testing and refactor||4/11 Checkpoint|
| 4/10     | Implement image classification model using layers we created           ||
| 4/17     | Parallel version CNN units implementation in CUDA         ||
| 4/24     | Compare implementation of CUDA, METAL and profiling, Report ||4/29 Report|
| 5/01     | Prepare for presentation ||5/05 Presentation|
