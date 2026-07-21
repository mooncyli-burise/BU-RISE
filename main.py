from simple_testing.simple_model_train_modified import train_simple
# from simple_testing.synthetic_real_limo_dataset.generate_limo_data import generate_synthetic_data
# from config import X_CLASSES, Y_CLASSES, ANGLE_CLASSES
from synthetic_limo_testing.synthetic_limo_dataset.generate_synthetic_data import generate_synthetic_dataset
# from synthetic_limo_testing.limo_train import train_limo
from util.visualize_simple import visualize

def main():
    # generate_synthetic_dataset()
    train_simple()
    visualize('/home/mooncyli/BU-RISE/simple_testing/simple_best_robot_detector.pth')
    print("done!")

if __name__ == "__main__":
    main()