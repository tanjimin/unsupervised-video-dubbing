
Project: Unsupervised Generative Video Dubbing

Authors: Jimin Tan, Chenqin Yang, Yakun Wang, Yash Deshpande

Project Advisor:  Prof. Kyunghyun Cho

## Introduction
We tackle the problem of generative video dubbing by modi-fying a video of a person speaking in one language so thatthe person is perceived as speaking the same content in an-other language.

**The goal of the project is to transform a video of a person speaking in one language so that it looks like the person is speaking the same content in another language.**
Given a video clip of a person reading a news article in language *S*, our model must produce a video clip of the same person reading the same news article however in language *T*.

<div style="text-align: center;">
<img src="assets/overall_structure.pdf" alt="Model structure" style="zoom:130%;" align="middle"/>
</div>  

#### Use case
- Fast, real-time translation of video broadcast content
- Detection of maliciously altered video content

## Dataset

**Data overview:**
- Video clips of people speaking 
	- Mouth position fixed in-frame
	- Initially used in an attempt to mislead a set of Visual Speech Recognition (VSR) systems 
- Dataset used:
	- Lip Reading Sentences in the Wild (http://www.robots.ox.ac.uk/~vgg/data/lip_reading/index.html#about)
	- Bloomberg newscast video

### Lip Reading in the Wild (LRW)

- Around 1000 utterances of 500 different words
- Each video consists of 29 frames (1.16 seconds) in length
- Audio has sampling rate of 16000 Hz
- Dataset size:
	- Training Set: 500 classes with 800-1000 utterances per class
	- Validation Set: 500 classes with 50 utterances per class
	- Test Set: 500 classes with 50 utterances per class

###Bloomberg newscast data

**Sample video**: 'People Actively Hate Us': Inside the Morale Crisis on the Border
- Content of video very different from content of LRW dataset (word-to-word comparison)
- Cropped (word-level) and pre-processed (resizing, face detection) word video for consistency
- Word - level time stamps extracted using AWS
- Max. length of phrase extracted from video that matches LRW words = 5 (large numbers of migrant families / gran número de familias migrantes)
- Facial landmarks extracted using DLib

<div style="text-align: center;">
<img src="assets/data_processing.pdf" alt="Model structure" style="zoom:130%;" align="middle"/>
</div>  
