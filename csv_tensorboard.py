import csv
from pathlib import Path

#*CSV
   
CSV_PATH = "results.csv"

def save_result(params, train_samples, accuracy, data_flag, model_name, NUM_EPOCHS, optimizer, BATCH_SIZE):
    file = Path(CSV_PATH)
    write_header = not file.exists()  # Header nur beim ersten Mal
    
    with open(file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["params", "train_samples", "accuracy", "information"])
        if write_header:
            writer.writeheader()
        writer.writerow({
            "params": params,
            "train_samples": train_samples,
            "accuracy": f"{accuracy:.3f}",
            "information": f"{data_flag, model_name, NUM_EPOCHS, optimizer, BATCH_SIZE}",
        })
