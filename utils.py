import torchvision.utils as vutils
import neptune


def save_image(x, name, it, filename, normalize=True):
    vutils.save_image(x, filename, normalize=normalize)
    neptune.send_image(name, x=it, y=filename)


def get_gin_params_as_dict(gin_config):
    params = {}
    for k, v in gin_config.items():
        for (kk, vv) in v.items():
            param_name = '.'.join(filter(None, k)) + '.' + kk
            param_value = vv
            params[param_name] = param_value

    return params