# Project: Weakly supervised learning-- label noise and correction


### [Full Project Description](doc/project3_desc.md)

Term: Fall 2022

+ Team 3
+ Team members
	+ Ying Gao 
	+ Alix Leon
	+ Shreya Sinha
	+ Weijia Wang
	+ Tomasz Wislicki

### Project summary: 

In this project, we built multiple image classifiers based on different architectures and got an accuracy of $\approx$ 96% at the end of the training for our final model.
There are 2 models in total. For model1, we tried a CNN model and several other pre-trained models including VGG16, ResNet18, MobileNet, AlexNet and EfficientNet. The the model with the performance was ResNet18, which got a 30% accuracy on a subset of the complete dataset, thus we chose it to be our model1. For model II, we extended the functionality of our Model I using the methods presented in [Inoue et al.](https://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w32/Inoue_Multi-Label_Fashion_Image_ICCV_2017_paper.pdf). We pretrained ResNet 18 on the clean labels and saved the weights for easy access. Afterwards, we added a label cleaning pipeline to learn a mapping from the noisy labels to the cleans labels. Finally, we extended our base CNN with two additional linear layers such that the model becomes sensitive to noisy labels as well as the clean labels. We tested the accuracy of Model II by manually checking the noisy data with our predictions from said model and concluded that the statistics were accurate with some margin of error.

### Preprocessing and Data Augmentation:
Before training the model, we first normalized the images and one-hot encoded the labels in our data. We also applied data augmentation using Keras' built-in data image generator to the training set by randomly rotating, flipping, and cropping the images. We also applied random erasing to the images to prevent overfitting.
	
**Contribution statement**: ([default](doc/a_note_on_contributions.md)) All team members contributed equally in all stages of this project. All team members approve our work presented in this GitHub repository including this contributions statement. 

```
proj/
├── lib/
├── data/
├── doc/
├── figs/
└── output/
```
