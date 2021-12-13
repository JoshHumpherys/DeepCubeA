import numpy as np
import pickle


def main():
    results = pickle.load(open("../data/cube3/test/data_0.pkl", "rb"))

    print(results.keys())
    # print([state.colors.tostring() for state in results['states']])
    print(["%i: %s" % (i, num) for i, num in enumerate(results['num_back_steps'])])

    # print(str([len(x) for x in results['solutions']]))
    # print([len(x) for x in results['solutions']])

    # lens1 = np.array([len(x) for x in results1["solutions"]])
    # lens2 = np.array([len(x) for x in results2["solutions"]])

    # print("%i states" % (len(results1["states"])))

    # print("\n--SOLUTION 1---")
    # print_results(results1)

    # print("\n--SOLUTION 2---")
    # print_results(results2)

    # print("\n\n------Solution 2 - Solution 1 Lengths-----")
    # print_stats(lens2 - lens1, hist=False)
    # print("%.2f%% soln2 equal to soln1" % (100 * np.mean(lens2 == lens1)))

    # lens = np.array([len(x) for x in results["solutions"]])

    # print("%i states" % (len(results["states"])))

    # print("\n--SOLUTION 1---")
    # print_results(results1)

    # print("\n--SOLUTION 2---")
    # print_results(results2)

    # print("\n\n------Solution 2 - Solution 1 Lengths-----")
    # print_stats(lens2 - lens1, hist=False)
    # print("%.2f%% soln2 equal to soln1" % (100 * np.mean(lens2 == lens1)))


if __name__ == "__main__":
    main()

