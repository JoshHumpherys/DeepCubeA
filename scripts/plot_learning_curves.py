from typing import List, Tuple

import matplotlib.pyplot as plt

from scripts.scrape_learning_curve import scrape_learning_curve


def plot_learning_curves(
        learning_curves: List[Tuple[List[int], List[float]]],
        plot_title: str,
        output_file_name: str) -> None:
    for learning_curve in learning_curves:
        plt.plot(learning_curve[0], learning_curve[1][:len(learning_curve[0])])
    plt.title(plot_title)
    plt.xlabel('Number of training states')
    plt.ylabel('%Solved')
    plt.legend(['small.json', 'medium.json', 'large.json', 'original'])
    plt.savefig(output_file_name)
    plt.show()


def main():
    back_steps = 12
    file_names_and_states_per_update = [
        ('../saved_models/cube3/cube3_2021-12-11T12-20-07/output.txt', 5000000),
        ('../saved_models/cube3/cube3_2021-12-12T11-09-50/output.txt', 5000000),
        ('../saved_models/cube3/cube3_2021-12-12T15-04-39/output.txt', 5000000),
        ('../saved_models/cube3/output.txt', 5000000),
    ]

    learning_curves: List[Tuple[List[int], List[float]]] = []

    for file_name, states_per_update in file_names_and_states_per_update:
        iteration_list, percent_solved_per_iteration_list, update_number_list, percent_solved_per_update_number_list =\
            scrape_learning_curve(back_steps, file_name)
        num_training_states_list = list(range(
            states_per_update,
            states_per_update * (len(percent_solved_per_iteration_list) + 1),
            states_per_update))
        learning_curves.append((num_training_states_list, percent_solved_per_iteration_list))
        # learning_curves.append((iteration_list, percent_solved_per_iteration_list))
        # learning_curves.append((update_number_list, percent_solved_per_update_number_list))

    plot_learning_curves(
        learning_curves,
        'Cube3 Learning Curve: (%%Solved using GBFS at %i back steps)\nvs number of training states' % back_steps,
        './output.png')


if __name__ == "__main__":
    main()

