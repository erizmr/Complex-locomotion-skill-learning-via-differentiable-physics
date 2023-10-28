import argparse


def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument('--config_file',
                        default='',
                        help='experiment config file')
    parser.add_argument(
        '--train',
        action='store_true',
        help='whether train model, default false')
    parser.add_argument(
        '--validate',
        action='store_true',
        help='whether validate model, default false')
    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='whether evaluate model, default false')
    parser.add_argument(
        '--no-tensorboard',
        action='store_true',
        help='disable tensorboard')
    parser.add_argument(
        '--evaluate_path',
        default='',
        help='the folder to validate')
    parser.add_argument(
        '--lr', type=float, default=2.5e-4, help='learning rate (default: 7e-4)')
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument(
        '--cuda-deterministic',
        action='store_true',
        default=False,
        help="sets flags for determinism when using CUDA (potentially slow!)")
    parser.add_argument(
        '--num-processes',
        type=int,
        default=1,
        help='how many training CPU processes to use (default: 16)')
    parser.add_argument(
        '--num-steps',
        type=int,
        default=1000,
        help='number of forward steps in A2C (default: 5)')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=1,
        help='log interval, one log per n updates (default: 10)')
    parser.add_argument(
        '--save-interval',
        type=int,
        default=50,
        help='save interval, one save per n updates (default: 100)')
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=None,
        help='eval interval, one eval per n updates (default: None)')
    parser.add_argument(
        '--num-env-steps',
        type=int,
        default=10e6,
        help='number of environment steps to train (default: 10e6)')
    parser.add_argument(
        '--env-name',
        default='SNMT',
        help='environment to train on')
    parser.add_argument(
        '--no-cuda',
        action='store_true',
        default=False,
        help='disables CUDA training')
    args = parser.parse_args()
    return args
