data_flag = 'pneumoniamnist' #channel = 3
NUM_EPOCHS = 1
BATCH_SIZE = 128
lr = 0.001
download = True
model_name = "SmallCNN"
size = 28 #Multiple Size Options: 28 (MNIST-Like), 64, 128, and 224


#medmnist_datasets = ['pneumoniamnist', 'breastmnist'] # Group 1 
#medmnist_datasets = ['tissuemnist', 'octmnist', 'organcmnist', 'organsmnist'] # Group 2
#medmnist_datasets = ['dermamnist', 'bloodmnist', 'pathmnist', 'retinamnist'] # Group 3
#medmnist_datasets = ['chestmnist'] # Group 4 
#medmnist_datasets = ['pneumoniamnist'] # Special case