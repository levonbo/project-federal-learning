import numpy as np

def calculate_distribution(num_clients, alpha, noniid=False):
    list = []
    if noniid == True:
        distribution = np.random.dirichlet(np.ones(num_clients) * alpha)
        for c in distribution: 
            list.append(round(float(c), 4))
    else:
        for c in range(num_clients):
            list.append(round(1/num_clients,4))
    return list