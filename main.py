from models import BasicCNN
import config
import dataset
import client
import server
import torch.nn as nn 


def main(seed):
    config.set_seed(seed)
    info, task, n_channels, n_classes,n_train_samples = config.get_info(config.data_flag)
    global_model = BasicCNN(in_channels=n_channels, num_classes=n_classes)
    rounds = 3
    client_loaders = dataset.get_client_loader(config.data_flag, 28, config.num_clients, config.batch_size)
    val_loader = dataset.get_val_loader(config.data_flag, 28, config.num_clients, config.batch_size)
    criterion = config.get_criterion(info)
    for r in range(rounds):
        print(f"Starting round {r+1}:…")
        client_weights = []
        client_sizes = [len(d) for d in client_loaders]

        # Train on each client
        for i, (tl, vl) in enumerate(zip(client_loaders, val_loader)):
            print(f"Client {i+1}")
            weights = client.train_local(global_model,tl, criterion, task, config.num_epoch)
            client_weights.append(weights)
            print(type(tl))
            print(type(vl))
            client.validate_model(global_model, vl, task, criterion, config.data_flag)


        # Normalize client sizes
        total_size = sum(client_sizes)
        normalized_weights = [s / total_size for s in client_sizes]

        # Aggregate
        global_weights = server.fedavg(client_weights, normalized_weights)
        global_model.load_state_dict(global_weights)
        print("FedAvg applied…")

        #client.validate_model(global_model,val_loader, task, criterion, config.data_flag )

                    
                    
if __name__ == "__main__":
    main(1)