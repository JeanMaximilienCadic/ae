import cv2

def format_img(img):
    from ae import cfg
    img = cv2.imread(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img, (cfg.img.W, cfg.img.W))
    img = img/255
    img *= 2
    img -= 1
    return img

def to_img(x):
    from ae import cfg
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, cfg.img.W, cfg.img.W)
    return x


