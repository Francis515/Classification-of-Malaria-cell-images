# Statistical Learning Course Project 

## Classification of Malaria cell images

### Methods

* Data reading and format transforming is implemented in Cell_image class.
* Image features extracted by SURF are used as X for training instead of the original images.
* The feature which representing the original image is selected based on the sum of the Harr wavelet response.
* Four classification methods are used: KNN, Random Forest, SVC, DNN.

![original image](pics/feature/f.png 'infected cell')
![features detected by SURF](pics/feature/allfeature_f.png 'features detected by SURF')
![key feature after selection](pics/feature/onefeature_f.png 'key feature after selection')

### Dataset

* The dataset is omitted. It can be downloaded from [kaggle datasets--cell images for detecting malaria](https://www.kaggle.com/iarunava/cell-images-for-detecting-malaria) and put it in project dictory. 
