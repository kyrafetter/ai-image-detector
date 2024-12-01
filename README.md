# AI-Image-Detector
UC San Diego CSE 151A: ML Learning Algorithms Final Project



# Link to jupyter notebook: <br>
https://github.com/kyrafetter/ai-image-detector/tree/Milestone4

# Milestone 4:

## Milestone Updates


* **Preprocessing- Guassian Noise:** For this model revision, we injected Gaussian Noise into all images (with a scaling factor of 0.1) in order to better mimic the noise present in natural image datasets and help our model become more robust and prevent overfitting. When this modification was exclusively tested on the 30 image dataset, it seemed to help improve model accuracy, increasing accuracy from 0.60 to 0.75. 

* **More Epochs:** For this model revision, we doubled the number of epochs from 5 to 10 in order to try and have more convergence in our loss, and achieve a higher accuracy, as recommended previously. While the model performance may still benefit from additional epochs- we haven't been able to run for additional epochs due to high runtime from computing hardware constraints- We did see some improvement in accuracy over last model.

* **Adaptive Gradient:** This model revision, we implemented an adaptive learning rate which seeks to help the model fine-tune its learning and avoid potential oscilating accuracy with higher epoch numbers. Given that our epoch count is relatively lower, we decided to halve the learning rate at each step, in order to not make the training too dramatically slow. If this model were to be run on a computing cluster or similar, with higher epoch counts, this adaptive learning rate could potentially be tuned to be a much sharper dropoff, albeit at later epochs. It is also worther mentioning, the lower epoch count is somewhat less concerning, given that our large training data set leads to 375 gradient updates per epoch or 3750 total across 10 epochs (as discussed in Milestone 3), so there are still a large number of gradient updates occuring, just not with thefull dataset.

* **Model Architecture:**

* **Other Exploration:** (kfold, etc)



## Milestone Questions

### 2. Evaluate your model and compare training vs. test error

Our model performs well currently, having the training error slightly higher than the test error. As discussed previously, we believe our large number of graident updates (375 per epoch) contributes to this.

### 3. Where does your model fit in the fitting graph? and What are the next models you are thinking of and why?

### 5. Conclusion section: What is the conclusion of your 2nd model? What can be done to possibly improve it? Note: The conclusion section should be it's own independent section. i.e. Methods: will have models 1 and 2 methods, Conclusion: will have models 1 and 2 results and discussion. 

### 6. Provide predictions of correct and FP and FN from your test dataset.


* Total Number of Testing Observations: 1200 
* True Positives: 517
* False Positives: 53
* True Negatives: 530
* False Negatives: 100
* Predictions can also be seen in the Milestone4-checkpoint.ipynb notebook.








# Milestone 3:

## Milestone Updates

* **Pre-Processing-Overall :** In regard to pre-processing, we have implemented the steps that we proposed in Milestone 2. This includes pixel normalization, image zero-padding and re-sizing, and image imputation.
* **Pre-Processing- Normalization:** For pixel normalization, we used Min-Max Normalization to normalize all pixel values to 0-1 from 0-255.
* **Pre-Processing- Padding:** In our project, we preprocess images by padding them with zeros to make them square before resizing, ensuring standardized dimensions. This approach preserves the original image values, as the padding does not alter the content of the images.
* **Pre-Processing- Imputation:** For image imputation, we used inpainting to impute regions of images that were corrupted/truncated.
* **Model Architecture:** Defined initial PyTorch model architecture and training, testing, and validation pipelines 5 epochs with a batch size of 32. Also added code for calculating metrics for precision, recall, F1, true positives, true negatives, false positives, false negatives, loss, accuracy, and graphs demonstrating changes in loss throughout model training and evaluation across iterations.
*  **Model testing:** Added plots of model accuracy and loss across the training and validation datasets. Also evaluated the training and test sets on the trained model and generated resultant performance metrics.



## Milestone Questions

### 3. Compare training vs. test error analysis

Generally, we see that our model performs well. This is due to the fact that we have 12,000 images and a batch size of 32, so we have 12,000/32 * 5 = 1875 gradient updates across the 5 epochs for our training data.



### 4. Answer the questions: Where does your model fit in the fitting graph? What are the next models you are thinking of and why?


We can see that the training has higher metric performance (please see above two cells for losses and metrics), however the testing metrics are also reasonably close and performing well. Due to this, we do not believe the risk of overfitting to be too high. Since we have other parameters we can modify (discussed below), we may still be able to decrease the loss of both training and test data sets. In other words, we may still have room to increase model complexity without overfitting.

In terms of next models, we may consider tweaking model parameters and structure (see more below).

### 6. Conclusion section: What is the conclusion of your 1st model? What can be done to possibly improve it?

We saw that our inital model appears to perform quite well. As stated above, to increase the performance, we will consider changing model architectural and runtime features for our next iteration. These improvements may include:


* **More epochs:** Training for more epochs would allow the model to continue learning, as the first couple of epochs will be needed just to reach a reasonable degree of accuracy, given the the model is randomly initialized. This is especially relevant because we chose to zero-pad non-square images so training for more epochs will enable the model to learn to ignore the zero-padding and to not use this feature when performing predictions.\
<br>

* **Adaptive learning rate:** Combined with a larger number of epochs especially, an adaptive learning rate (or at least one which changes throughout training a few times) may improve model performance. As the model gets increasingly accurate during epochs, a reduction in learning rate allows it to learn more slowly and not skip over a potential minimum.\
<br>


* **Increase training dataset size:** Right now we include 9600 images in our training dataset, and 1200 in our validation and test sets respectively, for a total of 12000 images across the three datasets. Because we have access to 60000 total images in our Kaggle dataset, we can increase the size of our train, validation, and test datasets to give the model more data to work with while training, and potentially increase it's ability to make more nuanced predicitions on unseen data.\
<br>

* **Randomize training data order:** We can randomize the order of the images in the training dataset, reshuffling between epochs, to avoid potential minor baises that may arise through the training data order.\
<br>

* **Model Architecture:** We plan on adding more convolutional layers to our model. Currently, we only have three convolutional layers that have feature maps of sizes 3, 32, and 64. We would like to add another convolutional layer for 128 feature maps so we can extract more abstract features from our images deeper in the model. We may also add model layers that consider brightness and sharpness of the images because in our exploratory data analysis, we observed variations in brightness and sharpness between AI generated and real images. We would like to distinguish these features further using model layers that address these features specifically.\
<br>

* **K-fold Cross Validation:** K-fold cross-validation helps improve our CNN model by providing a robust evaluation of its performance across different subsets of the data, ensuring it generalizes well to unseen images. By training on multiple folds and averaging results, it reduces the risk of overfitting or underfitting specific data splits, leading to more reliable model tuning and selection. \
<br>

* **Regularization:** We also plan to use regularization to help our model avoid overfitting. For example, we can use adversarial regularization to train the model not only with natural training data images but also perturbed data. By intentionally confusing the model with these perturbed images, we will encourage our model to become robust against adversarial perturbation and increase model performance.\
<br>







# Milestone 2: Preprocessing Data Plan: <br>

### 1 - Imputing Corrupt Image: <br>
We have one image in the set of real 6k images under the filepath "/Users/shrey/ai-image-detector/data/.ipynb_checkpoints/images/real/5879.jpg" <br>
that is corrupt due to being truncated halfway. We have three options for cleaning up this image. One option is to impute the image with another real image from online, another is to replace the image with another real image from the remaining images in the dataset because we are only using 12k of the 60k images, or remove the image from the dataset completely. We have decided to proceed with replacing the image with another real image from the portion of the kaggle dataset we are not using due to compute resource limiations. Taking this approach would allow us to maintain 6k images in the real images and 6k in the fake images, making the classes balanced. This would also ensure that the imputed image is not an outlier because the image would be coming from the same source dataset. <br>


### 2 - Resizing and Padding the Images: <br>
After performing a sizing analysis, we noticed our images are of varying non-square shapes. Hence, for batch processing/simultaneously processing groups of images in our CNN to work effectively, we need to resize our images to uniform size. <br>
a - Our largest images are in the 10000-14000 range, which are too large for a CNN with minimal computational capacity to process. Therefore, we will be replacing the largest outlier images with smaller images from the unused images in the remaining portion of the 60k dataset. <br>
b - We plan to resize our images to 1024 * 1024 pixels, so we will pad images smaller than this size to be 1024 * 1024 in order for these images to be used in batch processing with the remaining images. This could introduce noise to our CNN by having empty portions in our images, but this is the only process for us to maintain balance between the varying image dimensions in our dataset as we have both large and small images.  <br>
c - Any images that are non-square and larger than 1024 * 1024 will be resized to 1024 * 1024. This may cause some loss in image information, however, since our images are of varying sizes and tend to be relatively large, our only option is to resize the images to be smaller with the limited timeframe and compute resources we have for the execution of our project. Additionally, because the AI generated images tend to have color, sharpness, and brighntess differences from real images, we believe that loosing some image information should not prevent detection of these differentiating factors between images. <br>

* Note: We chose 1024 by 1024 because this is a common size that CNN kernels can handle effectively with limited computate resources and because a significant amount of our images cluser around the 2000 by 2000 pixel range. <br>


### 3 - Normalizing the Images RGB pixels: <br>

In our analysis, we noticed that our images have slight brightness, sharpness, and RGB differences between AI generated and real images. In order for these differences to be effectively and clearly differentiated by our CNN, we plan to normalize our RGB values to 0 - 1 rather than 0 - 255. We want to normalize the image size because this allows the model to learn from features that are at a consistent scale, improving its ability to generalize and recognize patterns. Normalizing RGB values also ensures that we work with smaller gradients during backpropagation allowing CNNs to train faster and more effectively as input data has a consistent and smaller scale. Finally, because we have brightness and color variations between AI generated and real images, normalizing the RGB values reduces the effect of these differences allowing the CNN to learn meaningful patterns along with intensity variations across the dataset rather than fully relying on brigthness and RGB. <br>

* Note: We chose normalization over standardization because normalized pixel valuesin the range [0, 1] are easier to interpret for the CNN as they align well with the original image structure. <br>


### 4 - Train Test Split Using Tensorflow ImageDataGenerator: <br>

We will use the Tensorflow ImageDataGenerator to perform our 80-20 train test split. The ImageDataGenerator will also be set with configurations for binary classfication, normalization, and 1024 by 1024 sizing (post padding the images), to streamline our data splitting and preprocessing steps. We will then build our CNN using Tensorflow. <br>

