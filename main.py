from simple_testing import simple_model_train
from simple_testing.synthetic_real_limo_dataset.generate_limo_data import generate_synthetic_data
from config import X_CLASSES, Y_CLASSES, ANGLE_CLASSES
from synthetic_limo_testing.synthetic_limo_dataset.generate_synthetic_data import generate_synthetic_dataset

def main():
    # generate_synthetic_data(X_CLASSES, Y_CLASSES, ANGLE_CLASSES)
    # generate_synthetic_dataset()
    simple_model_train.train_simple()
    print("done!")

if __name__ == "__main__":
    main()