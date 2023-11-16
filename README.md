# aitempo_stage_one_qual
Code used in the stage one qualification for the DTC. 

Running:
0) Setup:

  Create a Python virtual environment using the requirements.txt file 
  
  Get the testing and training data from the DTC (see: https://triagechallenge.darpa.mil/) 
  
1) Extract Features used for classification to a csv file
   
   $ python3 feature_extractor.py -a aortaP_train_data.csv -b brachP_train_data.csv
   
   $ python3 feature_extractor.py -a aortaP_test_data.csv -b brachP_test_data.csv
   
   This will produce the following two files: features_testing.csv and features_training.csv which are used in the classification script
   
 2) Train the classifier and use it to predict the age groups of the testing data
    
  $ python3 clasify.py --training features_training.csv --testing features_testing.csv

  This will output a json with the predicted classes for the testing data. 
    

