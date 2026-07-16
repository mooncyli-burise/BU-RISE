# from simple_testing import simple_model_train
from simple_testing.synthetic_real_limo_dataset.generate_limo_data import generate_synthetic_data
from config import X_CLASSES, Y_CLASSES, ANGLE_CLASSES

def main():
    generate_synthetic_data(X_CLASSES, Y_CLASSES, ANGLE_CLASSES)
    # simple_model_train.train_simple()
    print("done!")

if __name__ == "__main__":
    main()