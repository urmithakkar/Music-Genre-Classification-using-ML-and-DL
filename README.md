# **Music-Genre-Classification-using-ML-and-DL**


Music Genre Classification is a fundamental problem and is very challenging. Numerous models and algorithms have been developed over last few years to efficiently classify music according to its genres but some problems still exist. Various large datasets with Gigabytes of data are available for the task of music genre classification. However, because these datasets are huge, its a must to develop machine learning or deep learning models that are scalable and can classify music
information based on several important criterias such as genre,lyrics, instrument, artists etc.

Music classification tasks have two main challenges- (a) Proper pre-processing of the audio data and extracting all the relevant audio features required to classify the music. (b) Selecting the most efficient machine learning or deep learning model in order to get the desired results in music classification.

Over the past few years, deep neural networks have shown to produce outstanding results in various fields such as image processing, speech recognition, audio processing as well as natural language processing. Deep learning models like Convolutional Neural Network, Recurrent Neural Network and others have proven to be efficient with enormously large datasets, producing the desired accuracies in the results. While machine learning techniques suffer because of the large size
of the datasets and metadata, development of deep neural networks has proven to be very effective in dealing with the enormity of the datasets. Furthermore, in the domain of audio and music processing, deep neural networks have also shown the ability to perform improved audio feature extraction than the machine learning models, thus producing excellent results.

## Dataset

In this paper we have used the GTZAN Dataset for the task of classifying music into its genres. The dataset is divided into 10 genres, with 100 audio files for each genre. All of the audio clips have a length of 30 seconds. The audio clips are 22050 with mom 16 bit audio file in .wav format. The GTZAN dataset is the most widely used dataset for evaluation in music genre recognition(MGR).The dataset is divided into genres like pop, classical, hip-hop, Blues, Country, Disco, Jazz, Metal, Reggae and Rock


## Feature Extraction
Audio features are descriptions of sound. Different features capture different aspects of sound. 
These features play a major role in training machine learning algorithms to recognise patterns in order to solve a particular task.

### **Short Term Fourier Transform**

It is an important feature as it enables us to extract spectograms which we can then feed our neural network. Fourier transform is not performed on the whole audio signal, but rather we apply it on small segments of the audio signals.So, STFT is calculated for all the segments of the audio signal, this can be done by applying windowing to the signal. The windowing fucntion takes the original signal and then multiplies it by a window function sample by sample. Each chunk of the audio signal after windowing is called as a frame. Windowing of samples is done using three parameters, 
(i) window size: window size is the amount of samples that we apply windowing to. 
(ii)Frame size: Frame size refers to the number of samples that we consider in each chuck of a signal.
(iii) Hop size: Hop size tells us how many samples we slide to the right when we take a new chuck/frame. 

So STFT is applied to one chunk first and then it moves to the other until the end of the audio signal.

Now, the visualisation of STFT is done in form of spectrograms. Spectrograms have Time on the x-axis and Frequency on the y-axis. It shows how the different
frequency bins evolve over time across different frames present in the original signal. These spectrograms are then fed to our deep learning models to perform classification.

![image](https://github.com/urmithakkar/Music-Genre-Classification-using-ML-and-DL/assets/62324786/5f03cae5-e8c9-41e1-8d83-bf44807e197c)


### **Time domain featurres**

▶️ **Amplitude envelope:** It is the maximum value of all samples in a frame. It gives us a rough idea od the loudness. It is one of the features used for genre classification.


▶️ **Root-mean square energy:** it is the square root of mean of sum of energy for all the samples of a frame. It is the indicator of loudness and therefore is a great feature for music genre classificaation.


▶️ **Zero crossing rate:** its the number of times a signal crosses the horizontal axis.


▶️ **Mel-spectograms:** Mel spectograms are ideal for three reasons, time-frequency representation, human perceptual amplitude representation and human perceptual frequency representations. 
Humans percieve frequency/amplitude representations logarithmically, which is not possible to achieve with normal spectogram, therefore mel spectograms come in handy. 


▶️ **MFCCs:** MFccs are very good describers of music and capable of capturing the timbre. We usually take first 12-13 coefficients in considration because those are the most relevant 
ones. MFCCs are advantageous because they describe the large structures of the spectrum cutting down the noise that comes with the spectrum.


▶️ **Delta and delta-delta mfccs:** These are inshort the first and the second derivatives of the mfccs. They are very important because they tell us how the mfccs change over time in an audio file.
    
    
### **Frequency domain audio features:**

▶️ **Band energy ratio:** Band energy ratio is (the sum of power in lower frequencies) / (sum of the power of higher frequencies)
The split frequency gives us the threshold. All the frequencies above the threshold are higher frequencies and 
all the frequencies below the threshold are lower frequencies.It is extensively used in music genre classification.


▶️ **Spectral centroidd:** Spectral centroid is the weighted mean of the frequencies. It is one of the key frequency domain feature. It provides us with the center of gravity of the magnitude spectrum i.e. it gives us the frequency band which has most of the energy.
It measures the brightness of the sound as to how bright or dull a certain sound is.


▶️ **Bandwidth -** Bandwidth is related to the spectral centroid. It is the range which is of interest i.e the spectral range around the centroid. It is the weighted mean of the distances of the frequency bands from the spectral centroid. 
It is extensively used in traditional ML based music genre classification.


### **Rhythmic features**

▶️ **Tempo:** It is the estimation of music tempo to be used as a feature for audio classification. Tempo is one of the suitable features for audio genre classification because each genre will have a different playing speed.

### **Pitch content features**

▶️ **Chroma features:** These features give us more information about the notes in the music played. They can therefore be very useful in genre classification because songs of different genres would have different patterns of notes.


# Machine learning Architectures:


Here, I have used four machine learning models namely, SVM, Logistic Regresiion, Decission Tree Classifier and Random Forest Classifier. Each of these algorithms are frequently. used for audio classification.

Let's talk briefly about each model:

▶️ **Random Forest Classifier** The RandomForestClassifier from sklearn.ensemble was used for the random forest. It combines models and aims at producing a model that is better than any of the individual models. It uses decision trees to classify samples to the classes. 
In order to predict the class of a sample, a decision tree pass the value of features through a series of conditional statements. This model was chosen because it is easy to implement and tune 

▶️ **Logistic Regression:** It is used to understand the relationship between the dpepndent varaible and one or more independent variables by estimating properties using a logistic regression equation.

▶️ **Decision Tree Classifier:** It is a supervised machine learing model that uses a set of rules to make decisions.It is a distribution free and non parametric method which does not depend on probabily distribution assumptions.

▶️ **Support-Vector Machine:** The support vector machine was implemented using SVC from sklearn.svm. The hyperparameter 'C' is used to determine the level of regularization and the complexity of the model. It specifies the degree to which misclassified points should negatively impact the score of the model.
Kernel type- It determines the way in which the decision boundry of an SVM is drawn. 

# Deep learning Architectures:

▶️ **VGG-16 Architecture**

VGG16 (Simonyan & Zisserman 2014) is a convolutional neural network (CNN). A convolutional neural network is an artificial neural network which has an input layer, an output layer and a number of hidden layers. VGG16 is an object detection and classification algorithm which has an ability to classify 1000 images of 1000 different categories (Simonyan & Zisserman 2014). VGG16 has also shown promising results in the domain of audio
and music classification. The VGG16 architecture has 16 layers that have weights. It has thirteen convolutional layers, five max pooling layers and three dense layers.

The input to the architecture is a 224 x 224 image with 3 RGB channels. VGG16 is one of the best models because its focuses on small number of hyper-parameters like having convolutional layers of 3 x 3 filter with stride = 1 with same padding and max pooling layer of 2 x 2 filter with stride = 2. The first Conv layer (Conv1) has 64 filters, the second has 128 filters, the third has 256 filters, the fourth and fifth conv layers have 512 filters. This stack of convolutional layers is followed by three fully connected layers and the final layer is the Softmax layer (Simonyan & Zisserman 2014). In this paper the VGG16 model used was proposed by (Ahmad* & Sahil 2019). The activation function used is RELU activation function. In the last layer, we have used the softmax function which assigns probabilities to each class in a multi-class classification problem and these probabilities add up to 1.0.

![image](https://github.com/urmithakkar/Music-Genre-Classification-using-ML-and-DL/assets/62324786/d4d5057b-fcf2-44dc-8010-082cae65de6d)


# Conclusion and Future Work:

In this paper, we compared traditional machine learning algorithms with the popular deep neural network which is the VGG16 Architecture. We performed feature engineering by feeding some hand-crafted features to our machine learning models. The neural net was fed the spectrograms extracted using the Short-term-fourier-transform audio feature. Both the deep learning and machine learning are evaluated based on accuracy and confusion matrix representation is used to visualise the strengths and weaknesses of each model. To conclude, VGG16 architecture outperforms all the machine learning architectures with the final accuracy of 83% for 10 epochs on the test data. For future work, other neural networks can be implemented to produce improved classification of music genres. 2. VGG16 architecture has 138 million parameters which causes exploding gradient problem. Different ways can be studied to overcome VGG16’s exploding gradient problem and further study can be conducted. Due to lack of resources and time, we could not perform proper hyper parameter tuning, therefore by hyperparameter tuning, accuracy can be increased in future.


<img width="545" alt="image" src="https://github.com/urmithakkar/Music-Genre-Classification-using-ML-and-DL/assets/62324786/5507667d-917a-4642-9301-85142fcd47ea">


**In conclusion, for this dataset and problem statement, VGG-16 architecture proved to the best model as compared to the other models.**
