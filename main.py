from simple_testing import simple_model_train
from simple_testing.synthetic_real_limo_dataset.generate_limo_data import generate_synthetic_data

def main():
    simple_model_train.train_simple()
    # generate_synthetic_data()

if __name__ == "__main__":
    main()