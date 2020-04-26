
Project: Unsupervised Generative Video Dubbing

Authors: Jimin Tan, Chenqin Yang, Yakun Wang, Yash Deshpande

Project Advisor:  Prof. Kyunghyun Cho

## Introduction
We tackle the problem of generative video dubbing by modi-fying a video of a person speaking in one language so thatthe person is perceived as speaking the same content in an-other language.

**The goal of the project is to transform a video of a person speaking in one language so that it looks like the person is speaking the same content in another language.**
Given a video clip of a person reading a news article in language *S*, our model must produce a video clip of the same person reading the same news article however in language *T*.

Some of the benefit of solving this problem in the news cast area includes:
- Fast, real-time translation of video broadcast content
- Detection of maliciously altered video content


## Video Processing Pipeline
In this project, we propose a solution that is different from GANs to this generative problem. Notice that the core of this problem is to generate a series of mouth movements that is conditioned on the audio input and generating RGB pixels is not a crucial part of this task. Thus, we re-formulate this image generation problem to a regression problem by converting face RGB images to face landmark coordinates with the Dlib face recognition library. Then we only need to spend compuatation powers to generate mouth coordinates instead of pixel values. The abstraction from RGB pixels to coordinates simplifies the computation complexity by a large margin. Then the landmark to face process is out sourced to the Vid2Vid model from NVIDIA. Here we show a general pipeline for our solution.

<div style="text-align: center;">
<img src="assets/overall_structure.pdf" alt="Model structure" style="zoom:100%;" align="middle"/>
</div>  

## Data

### Dataset

- Video clips of people speaking with mouth position fixed in-frame
- Dataset used:
	- **Lip Reading Sentences in the Wild (LRW)** 
(http://www.robots.ox.ac.uk/~vgg/data/lip_reading/index.html#about)
	- **Bloomberg newscast video** (Internal Evaluation Dataset)

### Data Preprocessing
Since we re-formulated the problem to a regression problem, we need to transform the dataset to face landmarks using Dlib. We also transformed raw audio signals to MFCC features with the LibROSA library. The processed data is save in npz format.

<div style="text-align: center;">
<img src="assets/data_processing.pdf" alt="Data pre-processing" style="zoom:100%;" align="middle"/>
</div>  

## Model Strcuture
Our model adpoted an encoder-decoder design. We have two encoders for face and audio inputs respectively. The model output the landmark coordinates of the mouth area. All encoders and decoders consists of 3 feed-forward layers.

<div style="text-align: center;">
<img src="assets/model.pdf" alt="Model structure" style="zoom:100%;" align="middle"/>
</div>  

The input to our model includes the audio MFCC features and face landmark coordinates excluding the mouth area (As shown in the figure above). The output is the mouth landmark that is conditioned on both the face(mouth location) and audio features(mouth open status). 

In case that target audio signals are unavailable, we also build a character-level model where text inputs serve as an alternative of audio features. The major difference between the character encoder and the audio encoder is that the character-level text embeddings would be first up-sampled through a 1d transposed convolutional layer so that length of text context vector would match that of the frames.

## Model Training
We trained this model with our pre-processed data which includes face landmarks and MFCC audio features.

### Reconstruction Loss
Since we have the ground truth mouth landmark, the first part of the training is to train the network to align the mouth with ground truth given the two conditionals. We used MSE as the loss function for reconstruction.

<div style="text-align: center;">
<img src="assets/reconstruction.pdf" alt="Reconstruction" style="zoom:100%;" align="middle"/>
</div>  

### Contrastive Learning
Restruction loss with MSE has a natrual flaw that the loss function emphasize location correctness over open correctness since location difference usually takes a heavier toll on the loss function. Thus, we designed a loss function to emphasize open correctness. 

<div style="text-align: center;">
<img src="assets/loss_function_2.pdf" alt="Reconstruction" style="zoom:100%;" align="middle"/>
</div>  

The openness measurements(open level) mentioned above is defined as follows:

<div style="text-align: center;">
<img src="assets/loss_function.pdf" alt="Reconstruction" style="zoom:100%;" align="middle"/>
</div>  

The final training pipeline consists of reconsruction and contrastive learning. We combine them into one loss function at the end:

<div style="text-align: center;">
<img src="assets/training_graph.pdf" alt="Training" style="zoom:100%;" align="middle"/>
</div>  

After we trained this model, we can map a face landmark with corresponding audio directly into a mouth landmark.

## Post Processing

We use Vid2Vid from NVIDIA to convert generated face (with mouth) landmarks to images in RGB space. Then, we dynamically cropped out the mouth area and paste it to original background image.

<div style="text-align: center;">
<img src="assets/post_processing.pdf" alt="Post processing" style="zoom:100%;" align="middle"/>
</div>  

### Paste Smoothing 
The pasted mouth image patch can have boarders around it that looks unnatural. We applied a around smoothing technique that we demonstrated below:

<div style="text-align: center;">
<img src="assets/smoothing.pdf" alt="Post processing" style="zoom:100%;" align="middle"/>
</div> 


## Result

This is a 10-second video of a person originally speaking Spanish but conditioned on English audio.

<iframe src="https://drive.google.com/file/d/1tVfDd0cn6nh4w_KBQi8AXc8L8gQHuuhj/preview" width="640" height="480"></iframe>