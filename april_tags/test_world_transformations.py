from april_tags.get_data import get_apriltag_images
from april_tags.world_frame_transformations import get_world_coords, convert_to_cam

def test_world_transformations():
    tags = get_apriltag_images('april_tags')
    x, y = tags[0][0].center.astype(int)
    cam_pose = convert_to_cam(x, y)
    print("cam pose:", cam_pose)
    pose = get_world_coords(cam_pose)
    print("untransformed coords: ", f"({x}, {y})")
    print("pose: ", pose)