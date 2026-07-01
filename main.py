from models import BasicCNN
import config
import dataset
import client
import server
from datetime import datetime
import random   
from torch.utils.tensorboard import SummaryWriter
import numpy as np 
import copy


now = datetime.now()
uuid = random.randint(0,999)

def main(seed):
    writer = None
    if config.param["record_tensorboard"] == True: 
        run_name = f"{"fl"}__{uuid}_{config.param["data_flag"]}_{now:%Y-%m-%d__%H-%M}"
        writer = SummaryWriter(f"fl_tests_data_augmentation/{run_name}")
        writer.add_text("param", f"Optimizer: {config.param["optimizer"]} | Dataset: {config.param["data_flag"]} | Epochs: {config.param["num_epoch"]} | Batch Size: {config.param["batch_size"]} | lr: {config.param["lr"]} | Model: {config.param["model_name"]} | Data augmentation: {config.param["data_augmentation"]} ")
    config.set_seed(seed)
    info, task, n_channels, n_classes,n_train_samples = config.get_info(config.param["data_flag"])
    global_model = BasicCNN(in_channels=n_channels, num_classes=n_classes)
    client_loaders, distribution = dataset.get_client_loader(config.param["data_flag"], 28, config.param["num_clients"], config.param["batch_size"])
    val_loader = dataset.get_val_loader(config.param["data_flag"], 28, config.param["num_clients"], config.param["batch_size"])
    test_loader = dataset.get_test_loader(config.param["data_flag"], 28, config.param["num_clients"], config.param["batch_size"])
    criterion = config.get_criterion(info)
    config.print_intro(config.param["data_flag"],config.param["model_name"], uuid, config.param["num_clients"], distribution)
    for round in range(config.param["rounds"]):
        print(f"Starting round {round+1}:…")
        client_weights = []
        client_sizes = [len(d) for d in client_loaders]
        val_loss_per_client = []
        acc_per_client = []
        auc_per_client = []
        collection_train_loss_all_client = []
        # Train on each client
        for i, (tl, vl, testl) in enumerate(zip(client_loaders, val_loader, test_loader)):
            print(f"Client {i+1}")
            weights, avg_loss = client.train_local(global_model,tl, criterion, task, config.param["num_epoch"], config.param["lr"])
            client_weights.append(weights)
            collection_train_loss_all_client.append(avg_loss)

            local_model = copy.deepcopy(global_model)
            local_model.load_state_dict(weights)

            val_loss, acc, auc = client.validate_model(local_model, vl, task, criterion)
            print(f"Val -> Avg loss: {val_loss:.3f} - AUC: {auc:.3f} - Accuracy: {acc:.3f} ")
            val_loss_per_client.append(val_loss)
            acc_per_client.append(acc)
            auc_per_client.append(auc)
           

        if writer is not None:
            writer.add_scalars("Loss", {"train": np.average(collection_train_loss_all_client), "val": np.average(val_loss_per_client)}, round)
            writer.add_scalar("Val/acc per round", np.average(acc_per_client), round)
            writer.add_scalar("Val/auc per round", np.average(auc_per_client), round)

        # Normalize client sizes
        total_size = sum(client_sizes)
        normalized_weights = [s / total_size for s in client_sizes]

        # Aggregate
        global_weights = server.fedavg(client_weights, normalized_weights)
        global_model.load_state_dict(global_weights)
        print("FedAvg applied…")
        print("\n")

        if (round == config.param["rounds"] - 1 and writer is not None):
            print("\n")
            test_acc = client.test_model(global_model, testl, task, criterion) #type: ignore
            writer.add_scalar("Test/Test to parameters", test_acc, config.get_n_total_params(global_model))    
                    
if __name__ == "__main__":
    main(1)