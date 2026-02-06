import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    dataset_name = 'set2'

    parser.add_argument('--train_dataset',
                        default=f'./data/fusion/train/{dataset_name}.csv')

    parser.add_argument('--test_dataset',
                        default=f'./data/fusion/test/{dataset_name}.csv')

    parser.add_argument('--dim',
                        default=16544,
                        type=int)

    parser.add_argument('--ImbalancedDatasetSampler',
                        default=False,
                        type=bool)

    parser.add_argument('--epochs',
                        default=800,
                        type=int)

    parser.add_argument('--batchsize',
                        default=32,
                        type=int)

    parser.add_argument('--dropout',
                        default=0.5,
                        type=float)

    parser.add_argument('--lr',
                        default=0.00001,
                        type=float)

    parser.add_argument('--k',
                        default=3,
                        type=int)

    parser.add_argument('--embed_num',
                        default=16,
                        type=int)

    parser.add_argument('--cnn_kernel_size',
                        default=3,
                        type=int)

    parser.add_argument('--device',
                        default='cuda:0')

    parser.add_argument('--weight_decay',
                        default=1e-8,
                        type=float)

    parser.add_argument('--save',
                        default='checkpoints/')

    parser.add_argument('--test_batchsize',
                        default=256,
                        type=int)

    parser.add_argument('--cnn_out_channels',
                        default=16,
                        type=int)
    parser.add_argument('--cnn_stride',
                        default=2,
                        type=int)
    parser.add_argument('--kernel_size',
                        default=12,
                        type=int)

    parser.add_argument('--dropout_LC',
                        default=0.3,
                        type=float)
    parser.add_argument('--freeze_embed',
                        default=True,
                        type=bool)

    parser.add_argument('--stride',
                        default=3,
                        type=int)

    args = parser.parse_known_args()[0]
    return args
