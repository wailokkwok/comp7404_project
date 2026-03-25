# comp7404_project
This repo is a effort to replicate the results in the paper Accuracy, Interpretability, and Differential Privacy via Explainable Boosting

The code for DP-EBM is created by implementing the algorithm detials in the paper. The details implemented include quanitle binning, cyclic gradient boosting, 1-D greedy decision tree etc. 

The code for Logistic and Linear regression is implemted by using the diffprivlib package in IBM as instructed in the paper. 

The code for DP-Boost is implemented by using the following github link 
https://github.com/QinbinLi/DPBoost. 

To sucessfuly run the DP boost repository, our team have to upload the DPBOOST code to colab and run it from there due to incompatible cmake version of mac with the code. 

Because the health care and the wine dataset is not avaliable, we avoided it in the replcation of table 2 and 3. For figures 3,4 and 5, we use the adult income dataset to showcase the same results. 

To replicate experiment 1, run table gen for all methods. To replciate results of 2 and 3, run table_gen.py. 
