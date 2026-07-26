from april_tags.test_world_transformations import test_world_transformations, test_multiple_tags
from testing.detect import detect_predict
from backbone_model.real_world_dataset.generate_synthetic_data import generate_synthetic_dataset
from backbone_model.train_val_loop import train_real_world
from april_tags.test_detection import test_video_detection
from backbone_model import real_world_objects
from backbone_model import simple_model_objects_modified

from testing.visualize_real_world import visualize
from util.vid_to_image import extract_frames

from april_tags.create_ground_truth import create_ground_truth
from config import TEST_SIZE, TAG_SIZE_LIMO, APRILTAG_HEIGHT, APRILTAG_WIDTH

from util.print_apriltags import print_apriltags

def main():    
    # test_world_transformations()
    # test_multiple_tags()
    # test_video_detection()

    # generate_synthetic_dataset(5000)
    train_real_world(data_loader=simple_model_objects_modified.data_loader, data_loader_test=simple_model_objects_modified.data_loader_test, num_epochs=50, lr=1e-3, finetuning=False, checkpoint=False)
    # detect_predict('backbone_model/best_model_5000imgs.pth')

    # ground_truth_real_world = create_ground_truth('backbone_model/real_world_dataset/video', TAG_SIZE_LIMO)
    # print_apriltags('backbone_model/real_world_dataset/video', ground_truth_real_world)

    # create_ground_truth('backbone_model/real_world_dataset/video', 'backbone_model/real_world_dataset/video', TAG_SIZE_LIMO)
    # visualize('backbone_model/best_finetuning_model_lr1e-3.pth')

    

    # extract_frames("backbone_model/real_world_dataset/video/Untitled 2.mov", "backbone_model/real_world_dataset/video")

    print("done!")

if __name__ == "__main__":
    main()