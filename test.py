from april_tags.test_detection import test_image_detection, test_video_detection

#test_image_detection("/home/roboticslab/BU-RISE/april_tag_test_data")
test_video_detection()


# from dataset.epfl_processing import get_image_timestamp, calculate_car_orientations, get_centers, get_bounding_box

# print(get_bounding_box('/home/roboticslab/BU-RISE/data/epfl_dataset/tripod-seq/bbox_01.txt'))
# centers = get_centers('/home/roboticslab/BU-RISE/data/epfl_dataset/tripod-seq')
# print(centers)
# print(get_centers('/home/roboticslab/BU-RISE/data/epfl_dataset/tripod-seq'))


# print(get_image_timestamp('/home/roboticslab/BU-RISE/data/epfl_dataset/tripod-seq/tripod_seq_01_001.jpg'))
# print(calculate_car_orientations('/home/roboticslab/BU-RISE/data/epfl_dataset/tripod-seq'))

# from model.model import model
# import torch

# model.train()

# images = [torch.rand(3, 800, 800)]
# targets = [{
#     "boxes": torch.tensor([[100.,100.,120.,120.]]),
#     "labels": torch.tensor([1]),
#     "center": torch.tensor([110.,110.]),
#     "headings": torch.tensor([30]),
# }]

# losses = model(images, targets)

# print(losses)