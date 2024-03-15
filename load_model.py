import torch
import re


def my_load_weights(weight_path):

    print('Load checkpoint: %s' % weight_path)

    checkpoint = torch.load(weight_path, map_location='cuda')

    state_dict = {}

    for k, v in checkpoint['model'].items():

        # if k.startswith('conv_s8.'):
        #     continue
        # if k.startswith('upsample_s1.'):
        #     continue

        state_dict[k] = v

    return state_dict


def my_freeze_model(model):
    for name, param in model.named_parameters():
        pass
        # if name.startswith('upsample_s1.'):
        #     param.requires_grad = True
        # elif name.startswith('conv_s8.'):
        #     param.requires_grad = True
        # else:
        #     param.requires_grad = False