from backbone_model.simple_model_train_modified import train_simple
# from simple_testing.synthetic_real_limo_dataset.generate_limo_data import generate_synthetic_data
# from config import X_CLASSES, Y_CLASSES, ANGLE_CLASSES
from backbone_model.synthetic_limo_dataset.generate_synthetic_data import generate_synthetic_dataset
# from synthetic_limo_testing.limo_train import train_limo
from util.visualize_simple import visualize
from april_tags.test_world_transformations import test_world_transformations, test_multiple_tags
from util.detect import detect_predict

import cv2
from april_tags.world_frame_transformations import get_homography, show_homography_view

def main():
    # cap = cv2.VideoCapture(0)

    # H = get_homography()

    # while True:

    #     ret, frame = cap.read()

    #     if not ret:
    #         break

    #     show_homography_view(frame, H)

    #     if cv2.waitKey(1) == ord('q'):
    #         break

    # cap.release()
    # cv2.destroyAllWindows()
    
    # test_world_transformations()
    # test_multiple_tags()
    # generate_synthetic_dataset(5000)
    # train_simple()
    # visualize('backbone_model/best_model_5000imgs.pth')
    detect_predict('backbone_model/best_model_5000imgs.pth')

    print("done!")

if __name__ == "__main__":
    main()