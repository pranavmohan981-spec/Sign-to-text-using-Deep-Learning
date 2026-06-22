# Goal:Create 2 Models using Deep learning which can Convert Sign Language to text
This project develops two independent, deep-learning-based sign-to-text translation models to bridge communication gaps for the Deaf and hard-of-hearing communities. By splitting the system into static and dynamic recognition components, the architecture efficiently processes the distinct linguistic elements of American Sign Language (ASL).

Languages and Tools:
Python,Numpy,Tensor Flow,Open Cv,Media Pipe,Java

Model 1: ASL Alphabet Recognition
This model utilizes a Convolutional Neural Network (CNN) to recognize individual ASL hand signs from images or video frames.
We used the OpenCV (cv2) and operating system (os) libraries primarily for creating the dataset.Each letter is stored in a specific file named after the letter itself. We collected nearly 600 images for each alphabet
Our Model had 3 Convalutional Layer with 64,128,256 number of filters and 3 fully Connected layer with decreasing numbers of neurons (512, 256, 64) to progressively combine the extracted features from the convolutional layers and ReLU activation.
Realtime detection:Webcam stream is captured using OpenCV's video capture class and in that The ROI region is captured which is converted to grayscale and resized to 48X48 to fit the model requirements. The pre-processed image is then fed into loaded CNN model which predicts the probabilities of each possible sign language(A-Z), the predicted character with highest probability is defined using argmax function and its corresponding confidence score is displayed in the output frame.

<img width="671" height="278" alt="image" src="https://github.com/user-attachments/assets/f1d1b776-ab0a-4e6a-af0a-b27673a053df" />
