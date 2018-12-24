from __future__ import print_function

import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.backends.cudnn as cudnn
from data_loader import ImagerLoader, JustImagerLoader  # our data_loader
import numpy as np
from trijoint import im2recipe
import pickle
from args import get_parser

# =============================================================================
parser = get_parser()
opts = parser.parse_args()
# =============================================================================

torch.manual_seed(opts.seed)
if not opts.no_cuda:
    torch.cuda.manual_seed(opts.seed)


def main():
    model = im2recipe()
    model.visionMLP = torch.nn.DataParallel(model.visionMLP, device_ids=[0])
    # model.visionMLP = torch.nn.DataParallel(model.visionMLP, device_ids=[0,1])
    if not opts.no_cuda:
        model.cuda()

    print("=> loading checkpoint '{}'".format(opts.model_path))
    checkpoint = torch.load(opts.model_path)
    opts.start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}' (epoch {})"
          .format(opts.model_path, checkpoint['epoch']))

    # data preparation, loaders
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    img_loader = torch.utils.data.DataLoader(
        JustImagerLoader(opts.img_path,
                         transforms.Compose([
                             transforms.Scale(256),  # rescale the image keeping the original aspect ratio
                             transforms.CenterCrop(224),  # we get only the center of that rescaled
                             transforms.ToTensor(),
                             normalize,
                         ]),
                         data_path=opts.data_path),
        batch_size=1, shuffle=False,
        num_workers=opts.workers,
        pin_memory=(not opts.no_cuda)
    )

    # run test
    test(img_loader, model)


def test(img_loader, model):

    # switch to evaluate mode
    model.eval()

    for i, img in enumerate(img_loader):
        img = torch.autograd.Variable(img, volatile=True)
        img = img.cuda() if not opts.no_cuda else img.cpu()

        # compute output
        output = model(img, None, None, None, None)

        if i == 0:
            data0 = output[0].data.cpu().numpy()  # visual_emb
            data1 = output[1].data.cpu().numpy()  # visual_sem
        else:
            data0 = np.concatenate((data0, output[0].data.cpu().numpy()), axis=0)
            data1 = np.concatenate((data1, output[1].data.cpu().numpy()), axis=0)

    with open(opts.path_results + 'extrated_img_embeds.pkl', 'wb') as f:
        pickle.dump(data0, f)
    with open(opts.path_results + 'extract_img_sems.pkl', 'wb') as f:
        pickle.dump(data1, f)


if __name__ == '__main__':
    main()
