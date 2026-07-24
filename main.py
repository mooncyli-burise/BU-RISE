from backbone_model.simple_model_train_modified import train_simple
# from backbone_model.synthetic_limo_dataset.generate_synthetic_data import generate_synthetic_dataset
from util.visualize_simple import visualize
from april_tags.test_world_transformations import test_world_transformations, test_multiple_tags
from util.detect import detect_predict
from backbone_model.real_world_dataset.generate_synthetic_data import generate_synthetic_dataset
from backbone_model.real_world_train import train_real_world
from april_tags.test_detection import test_video_detection

def main():    
    # test_world_transformations()
    # test_multiple_tags()
    # test_video_detection()

    # generate_synthetic_dataset(5000)
    # train_simple()
    # visualize('backbone_model/best_model_5000imgs.pth')
    # detect_predict('backbone_model/best_model_5000imgs.pth')

    train_real_world()

    print("done!")

if __name__ == "__main__":
    main()