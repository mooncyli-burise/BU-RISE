from train_val_loop import train_real_world
import video_data_objects as vid
import real_world_objects as synthetic
from visualize_real_world import visualize
from real_world_dataset.generate_synthetic_data import generate_synthetic_dataset

def main():
    # train_real_world(vid.data_loader, vid.data_loader_test, 40, lr = 1e-3, finetuning = True, checkpoint = False, str = "_vid")
    # train_real_world(synthetic.data_loader, synthetic.data_loader_test, 40, lr = 1e-3, finetuning = True, checkpoint = False, str = "_synthetic")

    visualize("/home/mooncyli/BU-RISE/backbone_model/best_finetuning_model_synthetic.pth")
    # generate_synthetic_dataset(5000)

if __name__ == "__main__":
    main()