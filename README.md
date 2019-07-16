# Classification of Malaria cell images

## Feature based classification of Malaria cell images using KNN, Random Forest, SVC and DNN.

### Packages

Scikit-learn for KNN, Random Forest, SVC. 

Keras for DNN.

### Methods

* Data reading and format transforming is implemented in Cell_image class.
* Image features extracted by SURF are used as X for training instead of the original images.
* The feature which representing the original image is selected based on the sum of the Harr wavelet response.
* Four classification methods are used: KNN, Random Forest, SVC, DNN.


### Classification

![original image](pics/feature/f.png 'infected cell')
![features detected by SURF](pics/feature/allfeature_f.png 'features detected by SURF')
![key feature after selection](pics/feature/onefeature_f.png 'key feature after selection')

![original image](pics/feature/u.png 'infected cell')
![features detected by SURF](pics/feature/allfeature_u.png 'features detected by SURF')
![key feature after selection](pics/feature/onefeature_u.png 'key feature after selection')

* If the cell get infected, the selected feature will be located on the Malaria virus. If not infected, the selected feature will be on the edge.
* Use the discrepancy between two feature descriptors to realize classification.

### Feature selection

Max the sum of the Harr wavelet response

> ![](http://latex.codecogs.com/gif.latex?\\max\\sum_{f\\in}{(\\sum{|dx|_f}+\\sum{|dy|}_f)})

### Dataset

* The dataset is omitted. It can be downloaded from [kaggle dataset--cell images for detecting malaria](https://www.kaggle.com/iarunava/cell-images-for-detecting-malaria) and put it in project dictory. 
