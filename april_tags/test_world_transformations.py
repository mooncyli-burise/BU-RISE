from april_tags.get_data import get_apriltag_images
from april_tags.world_frame_transformations import get_world_coords

def test_world_transformations():
    tags = get_apriltag_images('april_tags')
    x, y = tags[0][0].center.astype(int)
    pose = get_world_coords(x,y)
    print("untransformed coords: ", f"({x}, {y})")
    print("pose: ", pose)