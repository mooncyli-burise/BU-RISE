from config import HEIGHT, WIDTH

def normalize_coords(x, y, neg_coords = False):
    if(neg_coords):
        x += WIDTH/2
        y += HEIGHT/2

    return x/WIDTH, y/HEIGHT