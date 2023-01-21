import math
import torch


def positionalencoding1d(d_model, length):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                         -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe


def positionalencoding2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    return pe

def positionalencoding3d(d_model, dx, dy, dz):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
#    if d_model % 6 != 0:
#        raise ValueError("Cannot use sin/cos positional encoding with "
#                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, dx, dy, dz)
    # Each dimension use half of d_model
    interval = int(d_model // 6) * 2
    div_term = torch.exp(torch.arange(0., interval, 2) * -(math.log(10000.0) / interval))
    pos_x = torch.arange(0., dx).unsqueeze(1) * div_term
    pos_y = torch.arange(0., dy).unsqueeze(1) * div_term
    pos_z = torch.arange(0., dz).unsqueeze(1) * div_term
    pe[0:interval:2, :, :, :] = torch.sin(pos_x).T.unsqueeze(2).unsqueeze(3).repeat(1, 1, dy, dz)
    pe[1:interval:2, :, :, :] = torch.cos(pos_x).T.unsqueeze(2).unsqueeze(3).repeat(1, 1, dy, dz)
    pe[interval:int(interval * 2):2, :, :] = torch.sin(pos_y).T.unsqueeze(1).unsqueeze(3).repeat(1, dx, 1, dz)
    pe[interval + 1:int(interval * 2):2, :, :] = torch.cos(pos_y).T.unsqueeze(1).unsqueeze(3).repeat(1, dx, 1, dz)
    pe[int(interval * 2):int(interval * 3):2, :, :] = torch.sin(pos_z).T.unsqueeze(1).unsqueeze(2).repeat(1, dx, dy, 1)
    pe[int(interval * 2) + 1:int(interval * 3):2, :, :] = torch.cos(pos_z).T.unsqueeze(1).unsqueeze(2).repeat(1, dx, dy, 1)

    return pe
