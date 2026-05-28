from datetime import date, datetime
from torch.utils.tensorboard import SummaryWriter
import dataset
import models
import config
import training
import testing
import csv_tensorboard



def main():
    for medmnist_dataset in config.param.data_flag:
        now = datetime.now()
        train_loader, train_loader_at_eval, test_loader = dataset.get_loader(medmnist_dataset, config.param.model_name, config.param.BATCH_SIZE, config.param.download, config.param.size)
        model = models.get_model(config.param.model_name, medmnist_dataset)
        optimizer = config.get_optimizer(config.param.optimizer, model, config.param.lr)
        criterion = config.get_criterion(medmnist_dataset)
        _, task, _, _, n_train_samples = config.get_info(medmnist_dataset)
        run_name = f"{medmnist_dataset}__{config.param.model_name}__{config.param.NUM_EPOCHS}__{now:%Y-%m-%d__%H-%M-%S}"
        writer = SummaryWriter(f"runs/{run_name}")
        print("Starting training...")
        for epoch in range(config.param.NUM_EPOCHS):
            print("Epoch",epoch+1)
            Aveg_loss = training.train_model(model, train_loader, optimizer, task, criterion)
            writer.add_scalar("Loss/train", Aveg_loss, epoch)

        print('==> Evaluating ...')
        train_split, train_metrics = testing.test('train', model, train_loader_at_eval, test_loader, task, medmnist_dataset)
        test_split, test_metrics = testing.test('test', model, train_loader_at_eval, test_loader, task, medmnist_dataset)
        train_auc, train_accuracy = train_metrics
        print(f"{train_split} AUC: {train_auc:.3f}, Accuracy: {train_accuracy:.3f}")

        test_auc, test_accuracy = test_metrics
        print(f"{test_split} AUC: {test_auc:.3f}, Accuracy: {test_accuracy:.3f}")

        if config.param.NUM_EPOCHS >= 2: 
            writer.add_text("param", f"Optimizer: {optimizer} | Dataset: {medmnist_dataset} | Epochs: {config.param.NUM_EPOCHS} | Batch Size: {config.param.BATCH_SIZE} | lr: {config.param.lr} | Model: {config.param.model_name} | Image size: {config.param.size} ")
            writer.add_scalar("Accuracy/ Parameters", test_accuracy, config.get_n_total_params(model))
            print("Results saved in Tensorboard!")
            csv_tensorboard.save_result(config.get_n_total_params(model), n_train_samples, test_accuracy, medmnist_dataset, config.param.model_name, config.param.NUM_EPOCHS, config.param.optimizer, config.param.BATCH_SIZE)
        else:   
            print("Results not saved in Tensorboard, due to not enough Epochs")
            print("\n")
        
if __name__ == "__main__":
    main()


