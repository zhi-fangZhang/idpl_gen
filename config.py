import argparse




def extract_args():
    # main setting
    parser = argparse.ArgumentParser(
        prog='IDGP demo file.',
        usage='Demo with partial labels.',
        epilog='end',
        add_help=True
    )
    # optional args
    parser.add_argument('--lr', help='optimizer\'s learning rate', type=float, default=1e-2)
    parser.add_argument('--wd', help='weight decay', type=float, default=1e-4)
    parser.add_argument('--lr_g', help='optimizer\'s learning rate', type=float, default=1e-2)
    parser.add_argument('--wd_g', help='weight decay', type=float, default=1e-4)
    parser.add_argument('--bs', help='batch size', type=int, default=256)
    parser.add_argument('--ep', help='number of epochs', type=int, default=500)
    parser.add_argument('--mo', type=str, default="resnet")
    parser.add_argument('--ds', help='specify a dataset', type=str, choices=['mnist', 'fmnist', 'kmnist', 'cifar10', 'cifar100', 'lost', 'MSRCv2', 'birdac', 'spd', 'LYN'],default='mnist')
    parser.add_argument('--rate', help='flipping probability', type=float, default=0.4)
    parser.add_argument('--warm_up', help='number of warm-up epochs', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0, required=False)
    # loss paramters
    parser.add_argument('--alpha','-alpha', type=float, default=1, help = 'balance parameter of the loss function (default=1.0)')
    parser.add_argument('--beta','-beta', type=float, default=1,help = 'balance parameter of the loss function (default=1.0)')
    parser.add_argument('--theta','-theta', type=float, default=1, help = 'balance parameter of the loss function (default=1.0)')
    parser.add_argument('--gamma','-gamma', type=float, default=1, help = 'balance parameter of the loss function (default=1.0)')
    parser.add_argument('--delta','-delta', type=float, default=1, help = 'balance parameter of the loss function (default=1.0)')
    parser.add_argument('--eta',  '-eta',   type=float, default=1, help = 'balance parameter of the loss function (default=1.0)')
    parser.add_argument('--T_1', type=float, default=1, help = 'balance parameter of the loss function (default=1.0)')
    parser.add_argument('--T_2', type=float, default=1, help = 'balance parameter of the loss function (default=1.0)')
    # model args
    parser.add_argument('--lo', type=str, default="idgp")
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    return args