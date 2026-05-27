from training import training
from testing import test
from config import n_total_params, writer, config, save_result, n_train_samples
from dataset import train_loader

def main():
    training()
    print('==> Evaluating ...')
    train_split, train_metrics = test('train')
    train_auc, train_accuracy = train_metrics
    print(f"{train_split} AUC: {train_auc:.3f}, Accuracy: {train_accuracy:.3f}")

    test_split, test_metrics = test('test')
    test_auc, test_accuracy = test_metrics
    print(f"{test_split} AUC: {test_auc:.3f}, Accuracy: {test_accuracy:.3f}")

    if config.NUM_EPOCHS >= 30: 
        writer.add_scalar("Accuracy/ Parameters", test_accuracy, n_total_params)
        print("Results saved in Tensorboard!")
        save_result(n_total_params, n_train_samples, ".3f" %test_accuracy)
    else:   
        print("Results not saved in Tensorboard, due to not enough Epochs")

if __name__ == "__main__":
    main()


