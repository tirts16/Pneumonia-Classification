# Pneumonia-Classification
Install ibm and create an environment and activate it.
Pull the PNEUMONIA_CLASSIFICATION folder.
Also pull the files from the ibmfl and perform the following.
->Replace keras_fl_model from your \anaconda3\envs\ibm\Lib\site-packages\ibmfl\model with the keras_fl_model from the model folder in ibml that was pulled from our repository. 
->Replace dataset from your \anaconda3\envs\ibm\Lib\site-packages\ibmfl\util with dataset from the util folder in ibml that was pulled from our repository.
->Add xray_keras_data_handler from util/data_handlers in ibmfl folder pulled from our repository to \anaconda3\envs\ibm\Lib\site-packages\ibmfl\util\data_handlers.

step 1:
Activate the ibm environment created  and go to the location where the PNEUMONIA_CLASSIFICATION folder is located.
//conda activate py36
//cd C:\Users\SAMISH\new\federated-learning-lib-main

step 2:
This command would generate 3 parties with 200 data points each, randomly sampled from the xray dataset. By default, the data is stored under the examples/data/xray/random directory.
python examples/generate_data.py -n 3 -d xray -pp 200

step 3:
This command specifies the machine learning model, here we have used Keras CNN classifier to train our model. It also generates the configuration files for training including the parties which are assumed to be joining the federated learning.Here, we use 3 parties for our dataset xray which needs to be mentioned using -d and the path examples/data/xray/random/ using -p.  
python  examples/generate_configs.py -n 3 -m keras_classifier -d xray -p examples/data/xray/random/

step 4:
To start the aggregator we need to be in the correct directory where our folder is present(examples) and run the aggregator configuration file. 
python -m ibmfl.aggregator.aggregator examples/configs/keras_classifier/config_agg.yml 
After the aggregator in initialised we have to start the aggregator by the following command.
START

step 5:
Open new terminal for each party and START and REGISTER each party.
cd C:\Users\SAMISH\new\federated-learning-lib-main
conda activate py36
python -m ibmfl.party.party examples/configs/keras_classifier/config_party0.yml
START
REGISTER

cd C:\Users\SAMISH\new\federated-learning-lib-main
conda activate py36
python -m ibmfl.party.party examples/configs/keras_classifier/config_party1.yml
START
REGISTER

cd C:\Users\SAMISH\new\federated-learning-lib-main
conda activate py36
python -m ibmfl.party.party examples/configs/keras_classifier/config_party2.yml
START
REGISTER

step 6:
After all the parties are registered open the aggregator terminal and train the model by the following command.
TRAIN

step 7:
After the completion of training of each party we have to evaluate the results by using the following command in the aggregator terminal.
EVAL

step 8:
To terminate the aggregator and the parties use the command STOP in the aggregator before exiting.
