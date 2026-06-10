import models
import config
import dataset
from client import Client
import torch.optim as optim
from medmnist import INFO
import torch.nn as nn 
import server


def main():
    global_model = models.get_model(config.model_name, config.data_flag)  
    global_model.train()
    info, task, n_channels, n_classes,n_train_samples = config.get_info(config.data_flag)

    #* Optimizer and criterion(loss function)
    optimizer = optim.Adam(global_model.parameters(), config.lr)
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

    for i in range(num_clients):
        client = Client(client_id=i,train_loader=client_loader[i], global_model=global_model)
        clients_list.append(client)

    for rds in range(config.rounds): 
        client_weights = []
        for ct_id in range(len(clients_list)):
            for epoch in range(config.num_epoch):
                weights = clients_list[ct_id].local_training(clients_list[ct_id].model, clients_list[ct_id].train_loader, optimizer, task, criterion)
                client_weights.append(weights)
            print(f"Client {clients_list[ct_id].client_id} finished training")

        #* FedAvg 
        global_weights = server.fedavg(client_weights, normalized_weights)
        global_model.load_state_dict(global_weights)
        #* Validation
        print(f"Round {rds+1} finished!")
        server.validate_model(global_model, val_loader, task, criterion, config.data_flag)
        print("\n")
                
                    
if __name__ == "__main__":
    main()