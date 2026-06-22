from models import BasicCNN
import config
import dataset
import client
import server
from datetime import datetime
import random   
from torch.utils.tensorboard import SummaryWriter
import numpy as np 


now = datetime.now()
record_tensorboard = True
uuid = random.randint(0,999)

def main(seed):
    writer = None
    if record_tensorboard == True: 
        run_name = f"{"fl"}__{uuid}_{config.data_flag}_{now:%Y-%m-%d__%H-%M}"
        writer = SummaryWriter(f"fl_tests/{run_name}")
    config.set_seed(seed)
    info, task, n_channels, n_classes,n_train_samples = config.get_info(config.data_flag)
    global_model = BasicCNN(in_channels=n_channels, num_classes=n_classes)
    client_loaders = dataset.get_client_loader(config.data_flag, 28, config.num_clients, config.batch_size)
    val_loader = dataset.get_val_loader(config.data_flag, 28, config.num_clients, config.batch_size)
    criterion = config.get_criterion(info)
    for round in range(config.rounds):
        print(f"Starting round {round+1}:…")
        client_weights = []
        client_sizes = [len(d) for d in client_loaders]
        val_loss_per_client = []
        acc_per_client = []
        auc_per_client = []
        # Train on each client
        for i, (tl, vl) in enumerate(zip(client_loaders, val_loader)):
            print(f"Client {i+1}")
            weights = client.train_local(global_model,tl, criterion, task, config.num_epoch)
            client_weights.append(weights)
            val_loss, acc, auc = client.validate_model(global_model, vl, task, criterion)
            print(f"Val -> Avg loss: {val_loss:.3f} - AUC: {auc:.3f} - Accuracy: {acc:.3f} ")
            val_loss_per_client.append(val_loss)
            acc_per_client.append(acc)
            auc_per_client.append(auc)
        
        
        if writer is not None:
            writer.add_scalar("Val/loss per round", np.average(val_loss_per_client), round)
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
                    
                    
if __name__ == "__main__":
    main(1)