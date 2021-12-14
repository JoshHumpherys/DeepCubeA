from argparse import ArgumentParser
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def main():
    # parse arguments
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help="")

    args = parser.parse_args()

    dataset = pickle.load((open(args.dataset, "rb")))

    path_length = dataset['solutions']

    plt.hist(path_length, weights=np.ones(len(path_length)) / len(path_length), bins=[0, 5, 10, 15, 20, 25, 30, 35, 40, 45])
    plt.xlabel('Distance from Goal State')
    plt.ylabel('Percentage of Scrambled States')

    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))

    fig = plt.gcf()
    cm = 1/2.54
    fig.set_size_inches(6, 3)
    plt.savefig('scripts/output_distribution.png')
    plt.show()

if __name__ == "__main__":
    main()