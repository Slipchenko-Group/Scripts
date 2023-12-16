In this directory you will find a few things:

2vtl_one.csv : this is just a dataset with one single molecule in it. Just for testing, if you want the full dataset, I can provide it

ANIEFP_model_forces_0BN0W_v2_21_best.pt : this is the .pt file that contains only the weights and biases of a trained ANI/EFP network

ANIEFP_model_forces_0BN0W_v2_preds.py : this is the file that I made to load a .pt file then run predictions on the dataset/single molecule, it is commented to describe how it works
if you need any more information on it please let me know

aep_calculator.py: this is where i took ANIs source code and edited a little to make the aev_computer into the aep_computer

efp_dataloader.py: this is the custom datasetclass i made for our datasets. This is just so we can easily load the .csv file and use it for trainig/predictions, the code is a bit messy, you probably dont need to look into it


