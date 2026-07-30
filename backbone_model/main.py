from train_val_loop import train_real_world
from video_data_objects import data_loader, data_loader_test

def main():
    train_real_world(data_loader, data_loader_test, 100, lr = 1e-3, finetuning = True, checkpoint = False)

if __name__ == "__main__":
    main()