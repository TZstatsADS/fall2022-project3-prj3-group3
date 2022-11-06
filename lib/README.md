### Code lib Folder

The lib directory contains various files with function definitions (but only function definitions - no code that actually runs).

lossfunction.py is the custom loss function for label cleaning method. We defined custom loss as l1-loss which wasnt predefined in tensorflow. In order to follow the paper we defined a custom loss function. This model is used to learn a mapping from the noisy labels to the clean labels from the validated dataset.The model is trained using an L1-loss function:
    $L = \sum_i |y_i - \hat{y_i}|$
where $y_i$ is the clean label and $y_hat_i$ is the predicted clean label. 

<br>


