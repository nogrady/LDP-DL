# Locally Differentially Private Distributed Deep Learning via Knowledge Distillation (LDP-DL)

- A preprint version of our paper: Link here

- This is a sample code of our LDP-DL paper, including all the necessary implementations: 1) DataPartation.py: split the data for data owners and data user; 2) TeacherModel.py: train the teacher models  using data owners' private data; 3) StudentModel_AQS.py: train the student model via active query sampling and knowledge distillation using data user's public data.   

- The splited data and trained teacher models are also provided. Due to the limitation of capacity, please download from this link:
 
#### Requirements

- Pytorch 1.4.0, numpy 1.18.1, scipy 1.4.1, cuda 10.0, joblib 0.14.1

#### Quick Start (only tested using PyCharm on Ubuntu, Python 3.7)  

- Download or clone the whole repository.  
- Import our code as a project into PyCharm.  
- Run DataPartation.py to split the data.
- Run TeacherModel.py to train teacher models.
- Run StudentModle_AQS.py to train and evaluate student model.

Contact info: zhuangdi1990@gmail.com, mingchenli1992@gmail.com
