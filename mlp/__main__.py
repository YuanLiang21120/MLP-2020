import argparse

from . import workout


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('task',type=str)
    parser.add_argument('--data', type=str, default='data')
    parser.add_argument('--tmp', type=str, default='tmp')
    parser.add_argument('--result', type=str, default='result')

    parser.add_argument('--train_size', type=int, default=10000)
    parser.add_argument('--test_size', type=int, default=2000)

    return parser.parse_args()


def main():
    args = _parse_args()
    workout.run(args)
    

if __name__ == '__main__':
    main()
