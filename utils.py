import torchvision.utils as vutils
import neptune


def save_image(x, name, it, filename, normalize=True):
    vutils.save_image(x, filename, normalize=normalize)
    neptune.send_image(name, x=it, y=filename)