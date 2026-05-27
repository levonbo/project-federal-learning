from training import training
from testing import test

def main():
    training()
    print('==> Evaluating ...')
    test('train')
    test('test')
    #print("Using", amount_total_params, "parameters")

if __name__ == "__main__":
    main()