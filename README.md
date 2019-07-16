# Classification of Malaria cell images

## Feature based classification of Malaria cell images using KNN, Random Forest, SVC and DNN.

### Packages

Scikit-learn for KNN, Random Forest, SVC. 

Keras for DNN.

### Methods

* Data reading and format transforming is implemented in Cell_image class.
* Since the traditional classification methods do not perform well in high dimension, the feature is used as a low dimension representation of the original image.
* Image features extracted by SURF are used as X for training instead of the original images.
* The feature which representing the original image is selected based on the sum of the Harr wavelet response.
* Four classification methods are used: KNN, Random Forest, SVC, DNN.

### Feature Selection

* Use SURF to detect features.
* Maximum the sum of the Harr wavelet response to select one feature.

> ![](http://latex.codecogs.com/gif.latex?\\max_{f}\\sum_{f\\in{F}}{(\\sum{|dx|_f}+\\sum{|dy|}_f)})

where F is the set of all features detected and f is one feature in the set F.

![original image](pics/feature/f.png 'infected cell')
![features detected by SURF](pics/feature/allfeature_f.png 'features detected by SURF')
![key feature after selection](pics/feature/onefeature_f.png 'key feature after selection')

![original image](pics/feature/u.png 'infected cell')
![features detected by SURF](pics/feature/allfeature_u.png 'features detected by SURF')
![key feature after selection](pics/feature/onefeature_u.png 'key feature after selection')

* If the cell get infected, the selected feature will be located on the Malaria virus. If not infected, the selected feature will be on the edge.
* Use the discrepancy between two feature descriptors to realize classification.

### Dataset

* The dataset is omitted. It can be downloaded from [kaggle dataset--cell images for detecting malaria](https://www.kaggle.com/iarunava/cell-images-for-detecting-malaria) and put it in project dictory. 
