# Bank-Note-Authentication-By-Using-MLP-Multi-Layer-Perception-
### Introduction
This is a Multilayer Perceptron neural network model for the banknote binary classification dataset.
This is just a simple example to demonstrate how a MLP model works .

### Dataset
<b> banknote authentication Data Set </b>

Source:
Volker Lohweg (University of Applied Sciences, Ostwestfalen-Lippe, volker.lohweg '@' hs-owl.de)

Data Set Information:

Data were extracted from images that were taken from genuine and forged banknote-like specimens. For digitization, an industrial camera usually used for print inspection was used. The final images have 400x 400 pixels. Due to the object lens and distance to the investigated object gray-scale pictures with a resolution of about 660 dpi were gained. Wavelet Transform tool were used to extract features from images.


Attribute Information:

1. variance of Wavelet Transformed image (continuous)
2. skewness of Wavelet Transformed image (continuous)
3. curtosis of Wavelet Transformed image (continuous)
4. entropy of image (continuous)
5. class (integer)

### Execution
Run main.py file for training and test set evaluation .
Here the optimizer is adam and loss function is 'Binary cross entropy' and train/test set is 0.66/0.33 .
Accuracy score is 0.989

