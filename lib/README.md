### Code lib Folder

The lib directory contains various files with function definitions (but only function definitions - no code that actually runs).

lossfunction.py is the custom loss function for label cleaning method. We defined custom loss as l1-loss which wasnt predefined in tensorflow. In order to follow the paper we defined a custom loss function. This model is used to learn a mapping from the noisy labels to the clean labels from the validated dataset.The model is trained using an L1-loss function:
    $L = \sum_i |y_i - \hat{y_i}|$
where $y_i$ is the clean label and $y_hat_i$ is the predicted clean label. 

<br>
model2.py : It is defining the architecture of image classifier and the label cleaning model. 
<br>
resnet.py : We have defined resnet-18.py architecture in the resnet.py file . Tensorflow dosent have a predefined resnet-18 model. Hence we defined its architecture to use it. Tensorflow only has a resnet-50 , however we didnt use it as it has many more parameters than resnet-50 which would make it more computationally expensive.

