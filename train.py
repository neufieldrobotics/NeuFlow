import torch
from torch.utils.data import DataLoader

import argparse
import os

from data_utils.datasets import build_train_dataset
from NeuFlow.neuflow import NeuFlow
from loss import flow_loss_func
from data_utils.evaluate import validate_things, validate_sintel, validate_kitti
from load_model import my_load_weights, my_freeze_model
from dist_utils import get_dist_info, init_dist, setup_for_distributed


def get_args_parser():
    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument('--checkpoint_dir', default=None, type=str)
    parser.add_argument('--dataset_dir', default=None, type=str)
    parser.add_argument('--stage', default='things', type=str)
    parser.add_argument('--val_dataset', default=['things', 'sintel'], type=str, nargs='+')

    # training
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--val_freq', default=1000, type=int)
    parser.add_argument('--num_steps', default=1000000, type=int)

    parser.add_argument('--max_flow', default=400, type=int)

    # resume pretrained model or resume training
    parser.add_argument('--resume', default=None, type=str)
    parser.add_argument('--strict_resume', action='store_true')

    # distributed training
    parser.add_argument('--local-rank', default=0, type=int)
    parser.add_argument('--distributed', action='store_true')

    return parser


def main(args):
    # torch.autograd.set_detect_anomaly(True)
    print('Use %d GPUs' % torch.cuda.device_count())
    # seed = args.seed
    # torch.manual_seed(seed)
    # np.random.seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = True

    if args.distributed:
        # adjust batch size for each gpu
        assert args.batch_size % torch.cuda.device_count() == 0
        args.batch_size = args.batch_size // torch.cuda.device_count()

        dist_params = dict(backend='nccl')
        init_dist('pytorch', **dist_params)
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        args.gpu_ids = range(world_size)
        device = torch.device('cuda:{}'.format(args.local_rank))

        setup_for_distributed(args.local_rank == 0)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # model
    model = NeuFlow().to(device)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank)
        model = model.module

    num_params = sum(p.numel() for p in model.parameters())
    print('Number of params:', num_params)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)

    start_step = 0

    if args.resume:

        state_dict = my_load_weights(args.resume)

        model.load_state_dict(state_dict, strict=args.strict_resume)

        my_freeze_model(model)

        for name, param in model.named_parameters():
            print(name, param.requires_grad)

        torch.save({
            'model': model.state_dict()
        }, os.path.join(args.checkpoint_dir, 'step_0.pth'))

    train_dataset = build_train_dataset(args.stage)
    print('Number of training images:', len(train_dataset))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=torch.cuda.device_count(),
            rank=args.local_rank)
    else:
        train_sampler = None

    shuffle = False if args.distributed else True
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               shuffle=shuffle, num_workers=args.num_workers,
                                               pin_memory=True, drop_last=True,
                                               sampler=train_sampler)

    # lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
    #     optimizer, args.lr,
    #     args.num_steps + 10,
    #     pct_start=0.05,
    #     cycle_momentum=False,
    #     anneal_strategy='cos',
    #     last_epoch=last_epoch,
    # )

    total_steps = 0
    epoch = 0

    counter = 0

    while total_steps < args.num_steps:
        model.train()

        # mannual change random seed for shuffling every epoch
        if args.distributed:
            train_sampler.set_epoch(epoch)

        for i, sample in enumerate(train_loader):
            img1, img2, flow_gt, valid = [x.to(device) for x in sample]

            model.init_bhw(img1.shape[0], img1.shape[-2], img1.shape[-1])

            flow_preds = model(img1, img2)

            loss, metrics = flow_loss_func(flow_preds, flow_gt, valid, args.max_flow)

            # more efficient zero_grad
            for param in model.parameters():
                param.grad = None

            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            # lr_scheduler.step()

            print(total_steps, metrics['epe'], metrics['mag'], optimizer.param_groups[-1]['lr'])

            total_steps += 1

            if total_steps % args.val_freq == 0:

                if args.local_rank == 0:
                    checkpoint_path = os.path.join(args.checkpoint_dir, 'step_%06d.pth' % total_steps)
                    torch.save({
                        'model': model.state_dict()
                    }, checkpoint_path)

                val_results = {}

                if 'things' in args.val_dataset:
                    test_results_dict = validate_things(model, dstype='frames_cleanpass', validate_subset=True, max_val_flow=args.max_flow)
                    if args.local_rank == 0:
                        val_results.update(test_results_dict)

                if 'sintel' in args.val_dataset:
                    test_results_dict = validate_sintel(model, dstype='final')
                    if args.local_rank == 0:
                        val_results.update(test_results_dict)

                if 'kitti' in args.val_dataset:
                    test_results_dict = validate_kitti(model)
                    if args.local_rank == 0:
                        val_results.update(test_results_dict)

                if args.local_rank == 0:

                    counter += 1

                    if counter >= 20:

                        for group in optimizer.param_groups:
                            group['lr'] *= 0.7

                        counter = 0

                    # Save validation results
                    val_file = os.path.join(args.checkpoint_dir, 'val_results.txt')
                    with open(val_file, 'a') as f:
                        f.write('step: %06d lr: %.6f\n' % (total_steps, optimizer.param_groups[-1]['lr']))

                        for k, v in val_results.items():
                            f.write("| %s: %.3f " % (k, v))

                        f.write('\n\n')

                model.train()

        epoch += 1


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    main(args)
