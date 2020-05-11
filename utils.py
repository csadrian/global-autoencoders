import torchvision.utils as vutils
import neptune


def save_image(x, name, it, filename):
    vutils.save_image(x, filename)
    neptune.send_image(name, x=it, y=filename)