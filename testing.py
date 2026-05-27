from config import model, task, config, writer, amount_total_params
import torch 
from dataset import train_loader_at_eval, test_loader
from medmnist import Evaluator

###*    Evaluation

def test(split):
    model.eval()
    y_true = torch.tensor([])
    y_score = torch.tensor([])
    
    data_loader = train_loader_at_eval if split == 'train' else test_loader

    with torch.no_grad():
        for inputs, targets in data_loader:
            outputs = model(inputs)

            if task == 'multi-label, binary-class':
                targets = targets.to(torch.float32)
                outputs = outputs.softmax(dim=-1)
            else:
                targets = targets.squeeze().long()
                outputs = outputs.softmax(dim=-1)
                targets = targets.float().view(-1, 1)

            y_true = torch.cat((y_true, targets), 0)
            y_score = torch.cat((y_score, outputs), 0)

        y_true = y_true.numpy()
        y_score = y_score.detach().numpy()
        
        evaluator = Evaluator(config.data_flag, split)
        metrics = evaluator.evaluate(y_score)
        _, accuracy = metrics
        print('%s  AUC: %.3f  accuracy:%.3f' % (split, *metrics))
        writer.add_scalar("Accuracy/ Parameters", accuracy, amount_total_params)
