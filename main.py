import models
import config
import dataset
from client import Client
import torch.nn as nn 
import server


def main():
    global_model = models.get_model(config.model_name, config.data_flag)  
    global_model.train()
    info, task, n_channels, n_classes,n_train_samples = config.get_info(config.data_flag)

    #* criterion(loss function)
    if info['task'] == "multi-label, binary-class":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()
    
    num_clients = 3
    client_loader = dataset.get_client_loader(config.data_flag, config.size,num_clients=num_clients, batch_size=config.batch_size)
    val_loader = dataset.get_val_loader(config.data_flag, config.size, config.batch_size)
    
    clients_list = []
    #* Right now: equal distribution
    normalized_weights = []
    for _ in range(num_clients):
        normalized_weights.append(float("%.2f" %(1/num_clients)))
    
    #* create clients 
    clients_list = [Client(client_id=i, train_loader=client_loader[i], global_model=global_model)for i in range(num_clients)]

    for rds in range(3): 
        print(f"Starting Round {rds+1}")

        #* Updating clients model
        for client in clients_list #!:
            client.model.load_state_dict(global_model.state_dict())

        client_weights = []
        #* Clients training
        for client in clients_list:
            print(f"Clients {client.client_id} training...")
            for epoch in range(config.num_epoch):
                client.local_training(client.train_loader, task, criterion)
            client_weights.append(client.model.state_dict())
            print(f"Client {client.client_id} finished training")

        #* FedAvg 
        global_weights = server.fedavg(client_weights, normalized_weights)
        global_model.load_state_dict(global_weights)
        print("New global model created using FedAvg...")
        #* Validation
        server.validate_model(global_model, val_loader, task, criterion, config.data_flag)

        print("\n")
                
                    
if __name__ == "__main__":
    main()