# Goal:Create 2 Models using Deep learning which can Convert Sign Language to text
This project develops two independent, deep-learning-based sign-to-text translation models to bridge communication gaps for the Deaf and hard-of-hearing communities. By splitting the system into static and dynamic recognition components, the architecture efficiently processes the distinct linguistic elements of American Sign Language (ASL).

Languages and Tools:

Python,Numpy,Tensor Flow,Open Cv,Media Pipe,Java

Model 1: ASL Alphabet Recognition

* This model utilizes a Convolutional Neural Network (CNN) to recognize individual ASL hand signs from images or video frames.

* We used the OpenCV (cv2) and operating system (os) libraries primarily for creating the dataset.Each letter is stored in a specific file named after the letter itself. We collected nearly 600 images for each alphabet

* Our Model had 3 Convalutional Layer with 64,128,256 number of filters and 3 fully Connected layer with decreasing numbers of neurons (512, 256, 64) to progressively combine the extracted features from the convolutional layers and ReLU activation.

* Realtime detection:Webcam stream is captured using OpenCV's video capture class and in that The ROI region is captured which is converted to grayscale and resized to 48X48 to fit the model requirements.The pre-processed image is then fed into loaded CNN model which predicts the probabilities of each possible sign language(A-Z), the predicted character with highest probability is defined using argmax function and its corresponding confidence score is displayed in the output frame.

<img width="671" height="278" alt="image" src="https://github.com/user-attachments/assets/f1d1b776-ab0a-4e6a-af0a-b27673a053df" />

* The model achieved a final training accuracy of 98.40%,This means that on the training set, the model correctly classified 98.40% of the sign language gestures. We have used the Categorical-cross entropy as the loss function the training loss of was almost 0.0554 average across all samples in batch or epoch.The model's performance on unseen data, evaluated using the validation set, is even more promising as it achieved a validation accuracy of 99%.


Model 2: Gesture Recognition
* This model employs a Long Short-Term Memory (LSTM) network in conjunction with MediaPipe for hand pose estimation to translate dynamic gestures into text

* We use a combination of media pipe and lstm for gesture classification, we import holistic module from mediapipe library this helps us to detect human pose landmarks, face landmarks and hand landmarks simultaneously.

* The mediapipe creates a NumPy array containing the x, y, z coordinates, and visibility information If there is no extracted data, then it creates a NumPy array of size then a NumPy array is created of size 33*4 () similar logic is applied for extracting key points of face, right hand and left hand and using this Array we are training out Model.

* This NumPy arrays (individual frame key points) for each sequence is transformed into a single sequence element within the sequences list. Each sequence element is a list containing the KeyPoint data for all frames within that sequence

* The first layer of model has input shape of 30(1 sequence) each element is a vector of 1662 dimensions. The first layer has 64 units which aids the complexity of the layer and its capacity to learn patterns within the data while the second layer has 128 and final LSTM layer has 64 units.

Fully Connected Layers

* The the first fully connected (dense) layer has 64 units. Dense layers transform the learned representation from the final LSTM layer into a format suitable for classification. The relu activation function is used for introducing non-linearity within the dense layer.

* Dense Layer-2 This is has 32 units. It further processes the output from the previous dense layer, potentially refining the representation for classification.

* Dense Layer-3 is the final output layer. The number of units in this layer is set to the number of sign language actions in our dataset.

Real Time Prediction

* We will set an empty list to store the sequence, as the webcam continuously captures once the sequence reaches 30 frames the model will analyse it to predict the most probable sign action that we have performed. Along with that we use coloured bar graphs to represent model’s confidence level of different signs on the screen

<img width="705" height="230" alt="image" src="https://github.com/user-attachments/assets/cbea5afd-65a1-4bdf-8746-2dfb830eb1c4" />

* The LSTM model achieved a final training accuracy of 95.87%, indicating a good fit on the training data,We have used the Categorical-cross entropy as the loss function the training loss of was 0.1260 average across all epochs suggesting model learned the patterns in the training data effectively. The model's performance on unseen data that is test data, achieved accuracy of 95%, whereas the loss was 0.2704
