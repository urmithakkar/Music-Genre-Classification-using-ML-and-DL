# **Music-Genre-Classification-using-ML-and-DL**


Using a part of **MLEnd Hums and Whistles dataset**, we implement ****Audio Genre Classification****. I have built a supervised machine learning pipeline that takes as an input audio files (starwars,Frozen,showman,Mamma,pink panther,hakuna matata) and predicts its genre (classical,show tune,Pop,jazz,reggae). The dataset consists of around 2360 audio files including six songs with their genres. We have compared machine learning models to automatically classify hums and whistles of songs to their genres. The topic is quite interesting because for humans, it is easy to recognise the genre of the song even if they are unfamiliar with it but in case of machines, it is not that easy. 

The question is how well can the machine predict the correct genre to the song provided. There are real world applications already that can predict Genres of the music using the lyrics of the songs. In our case the challenge is to predict the genres using hums and whistles and not the actual lyrics. We will be using the librosa package in python for music and audio analysis.It acts as a building block necessary to create music information retrieval systems.So let's jump into it.


## Machine Learning pipeline:

Generally, running machine learning algorithms involves sequence of tasks such as data pre-processing, dfeature extraction, fitting the model and performing validation task. A pipeline consists of sequence of tasks. Out of the two types of pipelines, Transformer and Estimator, we will be using an Estimator Pipeline. It will include stages like scaling the data and training the models on the dataset with labels and features. The pipeline takes the training dataset as an input, scales it, trains the model, predicts the genre of the song on the validation data and computes the accuracies as the output.


## Transformation stage
Audio features are descriptions of sound. Different features capture different aspects of sound. 
These features play a major role in training machine learning algorithms to recognise patterns in order to solve a particular task.

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


# Modelling


Here, I have used four machine learning models namely, Logistic Regresiion, Decission Tree Classifier and Random Forest Classifier. Each of these algorithms are frequently. used for audio classification.

Let's talk briefly about each model:

▶️ **Random Forest Classifier** The RandomForestClassifier from sklearn.ensemble was used for the random forest. It combines models and aims at producing a model that is better than any of the individual models. It uses decision trees to classify samples to the classes. 
In order to predict the class of a sample, a decision tree pass the value of features through a series of conditional statements. This model was chosen because it is easy to implement and tune 

▶️ **Logistic Regression:** It is used to understand the relationship between the dpepndent varaible and one or more independent variables by estimating properties using a logistic regression equation.

▶️ **Decision Tree Classifier:** It is a supervised machine learing model that uses a set of rules to make decisions.It is a distribution free and non parametric method which does not depend on probabily distribution assumptions.

▶️ **Support-Vector Machine:** The support vector machine was implemented using SVC from sklearn.svm. The hyperparameter 'C' is used to determine the level of regularization and the complexity of the model. It specifies the degree to which misclassified points should negatively impact the score of the model.
Kernel type- It determines the way in which the decision boundry of an SVM is drawn. 

▶️ **KNN:** It is a supervised machine learing model that classifies the samples to the same class that its k nearest neighbours belong to.


# Conclusions

his task included comparing machine learning algorithms as to how well they suited to the problem to Audio Genre Classification using the MLEnd Hums and Whistles dataset. Classification was based on several features extracted from each audio segment of the dataset. Analysis of accuracies and F-scores indicated that some of the features were more significant than the others. 

Several features were added and removed and experiments were carried out. However, for future improvements, several other features could be added and experimented with in terms of their relevance for this type of classification. Complex feature modifications in this model can also be carried out to obtain better results. Another way can be using other models and analysing the classification reports and confusion matrices.Overall, Audio Genre Classification is an interesting and worthwhile topic and has alot of scope for innovation. 

**In conclusion, for this dataset and problem statement, KNearestNeighbours proved to the best model as compared to the other models.**
