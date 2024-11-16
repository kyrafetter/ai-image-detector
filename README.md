# AI-Image-Detector
UC San Diego CSE 151A: ML Learning Algorithms Final Project



# Link to jupyter notebook: <br>
https://github.com/kyrafetter/ai-image-detector/blob/main/notebooks/.ipynb_checkpoints/Milestone2-checkpoint.ipynb <br>


## Milestone 3: 

### Model Fitting Graph Analysis

### Next Models



### Conclusions and Next Steps


***Potential improvements may include***:
* **More epochs:** Training for more epochs would allow the model to continue learning, as the first couple of epochs will be needed just to reach a reasonable degree of accuracy, given the the model is randomly initialized.
* **Adaptive learning rate:** Combined with a larger number of epochs especially, an adaptive learning rate (or at least one which changes throughout training a few times) may improve model performance. As the model gets increasingly accurate during epochs, a reduction in learning rate allows it to learn more slowly and not skip over a potential minimum.



## Milestone 2: Preprocessing Data Plan: <br>

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

