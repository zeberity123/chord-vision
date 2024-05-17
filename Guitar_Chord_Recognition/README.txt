This project runs on a Jupyter notebook. 

Folder structure: 
- Root: contains the ipynb file and the model exported during training
- chords: contain the images used when chords are classified to provide feedback to the user
- output: contains the images used to train the classifier

This project requires a camera to operate in real time. Adjust this code line with the index of the camera you are using for it to work correctly: 

vid = cv2.VideoCapture(1)
- where "1" is a zero-based index of cameras conected to your computer. 

This project requires the following libraries: 
- matplotlib
- OpenCV
- time
- os
- ipwidgets
- IPython
- numpy
- PyTorch
- torchvision
