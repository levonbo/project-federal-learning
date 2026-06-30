import random

def calculate_non_iid_distribution(num_clients):
    list = []
    sum = 0
    for n in range(num_clients-1):
        random_num = round(random.uniform(0, 0.4), 2)
        list.append(random_num)
        sum += random_num
    list.append(round(1-sum,2))
    return list