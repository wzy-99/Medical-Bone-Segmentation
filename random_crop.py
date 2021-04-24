import random


class RandomCrop:
    def __init__(self, min_ratio=0.6, max_ratio=1.0, safe=True):
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.safe = safe
    
    def random_region(self, width, height):
        ratio = random.random() * (self.max_ratio - self.min_ratio) + self.min_ratio

        w = int(width * ratio)
        h = int(height * ratio)

        if w == width - 1:
            left = 0
        else:
            left = random.randint(0, width - w - 1)
        if h == height - 1:
            top = 0
        else:
            top = random.randint(0, height - h - 1)
        
        return left, top, w, h

    def random_region_safe(self, width, height, box):
        x0, y0, x1, y1 = box
        bw, bh = x1 - x0, y1 - y0

        w_min_ratio = max(self.min_ratio, bw / width)
        w_max_ratio = self.max_ratio
        h_min_ratio = max(self.min_ratio, bh / height)
        h_max_ratio = self.max_ratio

        w_ratio = random.random() * (w_max_ratio - w_min_ratio) + w_min_ratio
        h_ratio = random.random() * (h_max_ratio - h_min_ratio) + h_min_ratio

        w = int(width * w_ratio)
        h = int(height * h_ratio)

        lmin = max(0, x1 - w)
        lmax = min(width - 1 - w, x0)
        tmin = max(0, y1 - h)
        tmax = min(height -1 - h, y0)

        left = random.randint(lmin, lmax)
        top = random.randint(tmin, tmax)
     
        return left, top, w, h

    def random_crop(self, image, box):
        if len(image.shape == 3):
            height, width, channel = image.shape
        else:
            height, width = image.shape
        
        if self.safe:
            left, top, w, h = self.random_region_safe(width, height, box)
        else:
            left, top, w, h = self.random_region(width, height)
        
        if len(image.shape == 3):
            cut = image[top:top + h, left:left + w, :]
        else:
            cut = image[top:top + h, left:left + w]
        return  cut, (left, top, w, h)        
