![cover_photo](https://hips.hearstapps.com/hmg-prod/images/indoor-plants-1-64f051a37d451.jpg?crop=1xw:0.9xh;center,top&resize=1200:*)
# Plant Leaf Diseases Diagnosis with Deep Learning

*Plant disease identification relies on the experience and expertise of farmers or gardeners. As a home gardener or houseplant hobbyist, most of the time the required experience and expertise is missing. A machine learning based plant leaf diseases identification tool becomes very helpful.*

*I have a firsthand experience where I helped my mother treat her ailing indoor plants. Unaware of the exact disease, my mother used several pesticides, which made everyone flee the house. It wasn't until we got advice from a plant expert, who suggested an appropriate treatment, that the flower began to recuperate. How I wish there was a plant disease identification tool.*

## 1. Data

In order to build a machine learning model for plant disease diagnosis, images of plant leaves with different diseases are required. After some search online, I found the following dataset on [Mendeley Data](https://data.mendeley.com/research-data/?) which was originally from a published journal paper.

* [Image Data](https://data.mendeley.com/datasets/tywbtsjrjv/1)

* [Journal Paper](https://www.sciencedirect.com/science/article/abs/pii/S0045790619300023?via%3Dihub)

## 2. Method

The problem at hand is to build a supervised classification model. This is a fundamental task in computer vision and several models have been developed for this purpose:

1. **Convolutional Neural Networks:** CNNs are the backbone of most modern image classification techniques. They can automatically learn the spatial features from images.

2. **CNN-based Models:** ResNet (Residual Networks) developed by Microsoft, GoogLeNet introduced by Google, VGGNet developed by Visual Graphics Group at Oxford and etc.

3. **Transformer-based Models for Vision:** Originally designed for natural language processing, transformers have been adapted to vision tasks with promising results.

In this work, a **9-layer CNN model** (structure below) was selected for the plant leaf disease diagnosis task. CNN models are the foundation of more advanced models and they are well developed and widely used in the deep learning community. 
![image](https://github.com/wangtuguahhh/Capstone_2/assets/130683390/84f30cc5-840f-4aa1-bb4a-40dc7187d4c7)



## 3. Data Challenges

After [wrangling the raw image data](https://github.com/wangtuguahhh/Capstone_2/blob/c367d1c0ef0730e7326bac6507f3ec7a4319484d/notebook/Capstone2_01_Data_Wrangling.ipynb), there are 39 classes of images with very **imbalanced number of samples in each class**. 

Here are the numbers of images in each class for the training data and testing data.

![image](https://github.com/wangtuguahhh/Capstone_2/assets/130683390/6ad33ab4-35af-43a9-adee-5d895a9d5549)

![image](https://github.com/wangtuguahhh/Capstone_2/assets/130683390/27582da8-5c1f-4cb0-b701-c1607478e387)

By carefully selecting model metrics, the imbalanced testing data may not be a big issue. However, the imbalanced training data centainly will have negative impacts on model building.

Besides the imbalanced class issue, there are several classes with a very **limited number of images**, the smallest number only 122. With such limited data, it is hard for any model to give accurate predictions.

To address the limiting sample issue and imbalanced class issue in the training data:
* **Solution 1:** The ideal approach will be collecting more data for the classes with limited amounts of samples. In this work, inspired by the original journal paper, **data augmentation** using existing data was implemented to increase number of samples.

* **Solution 2:** With unbalanced data, ituitively we can down-scale or up-scale the data. The widely used solution in practice is **Boostrapping**, which mitigates the data unbalance issue and improves model accuracy. Therefore, boostrapping was tried in this work. 

[Data Augmentation Notebook](https://github.com/wangtuguahhh/Capstone_2/blob/c367d1c0ef0730e7326bac6507f3ec7a4319484d/notebook/Capstone2_01_Data_Wrangling.ipynb)

[Boostrapping Notebook](https://github.com/wangtuguahhh/Capstone_2/blob/c367d1c0ef0730e7326bac6507f3ec7a4319484d/notebook/Capstone2_03_Feature_Engineering.ipynb)

## 4. Simple Pre-processing

Pre-processing on image data was kept simple for this work. Try to decrease the computational cost but maintain the key features in the original data as much as possible.

Here is the step-by-step pre-processing on the input data:

![image](https://github.com/wangtuguahhh/Capstone_2/assets/130683390/0e9f2bdf-2109-4f67-ab8d-be8b8f61e7c3)

*Note that color images were used for modeling in this work. Gray scale is evaluated but not tried for modeling yet.*

[Pre-processing Notebook](https://github.com/wangtuguahhh/Capstone_2/blob/c367d1c0ef0730e7326bac6507f3ec7a4319484d/notebook/Capstone2_02_Data_Preprocessing_EDA.ipynb)

## 5. Modeling and Evaluation

The CNN model was built using [TensorFlow.Keras.Models](https://www.tensorflow.org/api_docs/python/tf/keras/Sequential). Model structure and parameters were kept simple and some from the journal paper. 

The focus was to test performance of models built from the following approaches:

![image](https://github.com/wangtuguahhh/Capstone_2/assets/130683390/74ed6db0-1f62-42d2-9029-6460cef81862)


Accuracy, cross_entropy, classification report and confusion matrix were selected as the metrics for model evaluation.

![image](https://github.com/wangtuguahhh/Capstone_2/assets/130683390/04ceb56b-48cf-4c4f-9b41-a74ab6fb091d)

*Note that for model_3 and model_4, a majority vote was used to generate the final output from the 5 different Bootstrapping models.*

![image](https://github.com/wangtuguahhh/Capstone_2/assets/130683390/836ffe05-2be7-4395-b107-d4bb01fa63d3)

![image](https://github.com/wangtuguahhh/Capstone_2/assets/130683390/c6a88b2b-9a40-4ebc-9b27-9042b11b0fa0)

**WINNER: model_1 & model_4**

model_1 had the best image quality since there was no data augmentation. model_4 had the best balanced classes with adquate samples. Therefore, both data quality and quantity play an important role on how good the model is.

[Modeling Notebook](https://github.com/wangtuguahhh/Capstone_2/blob/c367d1c0ef0730e7326bac6507f3ec7a4319484d/notebook/Capstone2_04_Modeling.ipynb)

## 6. Hyper-parameter Tuning

The original CNN model was tuned using the hyperband optimization algorithm to improve model performance with approach 4. The hyperband optimization better utilized the resources in a more efficient way compared to grid search or random search. `kears_tuner.Hyperband` packadge was used to implement the hyperband optimization. The following parameters were tested:
* filter numbers in the 2nd Conv2d
* filter numbers in the 3rd Conv2d
* rate in Dropout layer
* optimizer method in model.compile

Here is the optimized CNN structure.
![image](https://github.com/wangtuguahhh/Capstone_2/assets/130683390/49805804-9626-4e07-be1c-f73e10b5c85a)

The tuned model provided better predictions on test data. Compared to the original setup, more filter numbers in the last convolution layer and less dropout were prefered by the hyperband optimization algorithm.

![image](https://github.com/wangtuguahhh/Capstone_2/assets/130683390/87a75099-ae06-46a5-9e9a-5bdc586a8576)

[Hyper-parameter tuning notebook](https://github.com/wangtuguahhh/Capstone_2/blob/77615176f66456ff387273edaffc2fa00dc480b3/notebook/Capstone2_05_Modeling_Hyperparameter_Tuning.ipynb)

## 7. Future Improvements

There is so much more to improve for this task. 

**Modeling:**

Model hyper parameter tuning, such as more filters in convolution layers, more units in the fully connected layer, learning rate in optimizer, epoch numbers and etc.

**Feature Engineering:**

* Bootstrapping to draw more images from data without augmentation, such as to draw 500 images for each class.
* If improvement is observed for the step above, try drawing more images for each class to find the sweet spot.
* Bootstrapping to draw more images from data with augmentation, say 1500 for each class to see if there are any benefits.
* Find a balance between bootstrapping sample size the ensembling times, small sample size requiring more ensembling times.

**Pre-processing:**

After settling down on the model and feature engineering part, try using gray scale images as the input to reduce computational cost.

**Data Augmentation:**

* Multiple data augmentation methods were used in this work, such as rotation, flip, scaling, adding noises, etc. Try to limit to one augmentation method to see if certain augmentation improves model performance.
* Explore other augmentation techniques.

**Input Data:**

Search online to collect more images for the classes with a limited number of samples.

## 8. Acknowledgement

Thanks to Nicolas Renotte for his fun and engaging YouTube Video on [Build a Deep CNN Image Classifier](https://youtu.be/jztwpsIzEGc?si=jloqKAHLX2557qRR) and Raghunandan Patthar for being a super supporting Springboard mentor.




