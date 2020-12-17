# AMLS_assignment

## Applied Machine Learning System (ELEC0134) Assignment

Student Number: 17027443

Required Python3 

**Environment configuration**

OS: Windows10

bash command:

* conda create -n amls-tf2.0 python=3.7.6

* conda activate amls-tf2.0

* pip install Matplotlib==3.1.2

  pip install dlib==19.19.0

  pip install keras==2.3.1
  pip install keras-Applications==1.0.8
  pip install keras-Preprocessing==1.1.0
  pip install numpy==1.18.0

  pip install pandas==0.25.3

  pip install tensorflow==2.0.0

  pip install scikit-learn==0.22.1
  
  pip install opencv-python

**File Function**

**preprocess.py**

extract training ,validation and test data for task A1, A2, B1 and B2

**params.py**

Contains facial landmarks extraction function for task A1 and A2.

Includes several parameters for dataset path and filename

**shape_predictor_68_face_landmarks.dat**

Face recognition 68 feature points detection database

**main.py**

main file to execute in bash.