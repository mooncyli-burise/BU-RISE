from april_tags.get_data import get_apriltag_by_image
from april_tags.create_ground_truth import create_ground_truth_vid

class AprilTag:
    def get_ground_truth(self, frame):
        tags = get_apriltag_by_image(frame)
        ground_truth = create_ground_truth_vid(tags)
        return ground_truth