import numpy as np
from localization.world_frame import WorldFrame

world_frame = WorldFrame()

pixel = np.array([350, 200])

world = world_frame.pixel_to_world(pixel)

pixel2 = world_frame.world_to_pixel(world)

print(pixel)
print(pixel2)