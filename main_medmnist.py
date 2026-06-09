from datetime import date, datetime
from torch.utils.tensorboard import SummaryWriter
import dataset
import models
import config
import training
import testing
import validation
import csv_tensorboard
import random
import numpy as np 
import torch
import matplotlib.pyplot as plt

now = datetime.now()
record_tensorboard = True

def main(seed):
    for medmnist_dataset in config.param.data_flag:
        #* Seed setting
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        train_loader, val_loader, test_loader = dataset.get_loader(medmnist_dataset, config.param.model_name, config.param.BATCH_SIZE, config.param.download, config.param.size)
        model = models.get_model(config.param.model_name, medmnist_dataset)
        optimizer = config.get_optimizer(config.param.optimizer, model, config.param.lr)
        criterion = config.get_criterion(medmnist_dataset)
        _, task, _, _, n_train_samples = config.get_info(medmnist_dataset)
        writer = None
        if record_tensorboard == True: 
            run_name = f"{medmnist_dataset}__{config.param.model_name}__{config.param.NUM_EPOCHS}__{now:%Y-%m-%d__%H-%M-%S}"
            writer = SummaryWriter(f"runs/{run_name}")
        #* -> Training
        print("Starting training...") 
        training_loss = []
        validation_loss = []
        for epoch in range(config.param.NUM_EPOCHS):
            print(f"Epoch {epoch+1}")

            train_loss = training.train_model(model, train_loader, optimizer, task, criterion)
            val_loss,auc,acc = validation.validate_model(model, val_loader, task, criterion, medmnist_dataset)

            training_loss.append(train_loss)
            validation_loss.append(val_loss)
            if writer is not None:
                writer.add_scalar("Train/Loss", train_loss, epoch)
                writer.add_scalar("Validation/Loss", val_loss, epoch)
                writer.add_scalar("Validation/AUC", auc, epoch)
                writer.add_scalar("Validation/Accuracy", acc, epoch)

        plt.plot(training_loss, label="Training")
        plt.plot(validation_loss, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()
        print('==> Evaluating ...')
        test_split, test_metrics = testing.test('test', model, test_loader, task, medmnist_dataset)
        test_auc, test_accuracy = test_metrics
        print(f"{test_split} AUC: {test_auc:.3f}, Accuracy: {test_accuracy:.3f}")

        if writer is not None:
            writer.add_text("param", f"Optimizer: {optimizer} | Dataset: {medmnist_dataset} | Epochs: {config.param.NUM_EPOCHS} | Batch Size: {config.param.BATCH_SIZE} | lr: {config.param.lr} | Model: {config.param.model_name} | Image size: {config.param.size} ")
            writer.add_scalar("Accuracy/ Parameters", test_accuracy, config.get_n_total_params(model))
            print("Results saved in Tensorboard!")
            csv_tensorboard.save_result(config.get_n_total_params(model), n_train_samples, test_accuracy, medmnist_dataset, config.param.model_name, config.param.NUM_EPOCHS, config.param.optimizer, config.param.BATCH_SIZE)
        else:   
            print("Results not saved in Tensorboard")
            print("\n")
        plt.show()
        
if __name__ == "__main__":
    main(seed=42)