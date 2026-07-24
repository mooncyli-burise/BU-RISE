from config import HEIGHT, WIDTH

def normalize_coords(x, y, neg_coords = False, width = WIDTH, height = HEIGHT):
    if(neg_coords):
        x += width/2
        y += height/2

    return x/width, y/height