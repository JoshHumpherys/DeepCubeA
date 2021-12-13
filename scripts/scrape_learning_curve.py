import re
from typing import List, Tuple


def scrape_learning_curve(back_steps: int, file_name: str) -> Tuple[List[int], List[float], List[int], List[float]]:
    with open(file_name) as file:
        output = file.readlines()

    update_number_regex = re.compile('Training model for update number (\d*) for (\d*) iterations')
    percent_solved_regex = re.compile('Back Steps: %i, %%Solved: (\d*\.?\d*),' % back_steps)

    total_num_iterations = 0
    iteration_list: List[int] = []
    percent_solved_per_iteration_list: List[float] = []

    update_number_list: List[int] = []
    percent_solved_per_update_number_list: List[float] = []

    duplicate_update_number = False
    for line in output:
        match = update_number_regex.match(line)
        if match is not None:
            total_num_iterations += int(match.group(2))
            iteration_list.append(total_num_iterations)

            if len(update_number_list) > 0 and update_number_list[-1] == match.group(1):
                duplicate_update_number = True
            else:
                update_number_list.append(int(match.group(1)))

        match = percent_solved_regex.match(line)
        if match is not None:
            percent_solved_per_iteration_list.append(float(match.group(1)))

            if duplicate_update_number and len(percent_solved_per_update_number_list) > 0:
                percent_solved_per_update_number_list[-1] = float(match.group(1))
                duplicate_update_number = False
            else:
                percent_solved_per_update_number_list.append(float(match.group(1)))

    return iteration_list, percent_solved_per_iteration_list, update_number_list, percent_solved_per_update_number_list


def main():
    back_steps = 20
    file_name = '../saved_models/cube3/cube3_2021-12-11T12-20-07/output.txt'

    iteration_list, percent_solved_per_iteration_list, update_number_list, percent_solved_per_update_number_list =\
        scrape_learning_curve(back_steps, file_name)

    print([x for x in zip(iteration_list, percent_solved_per_iteration_list)])
    print([x for x in zip(update_number_list, percent_solved_per_update_number_list)])


if __name__ == "__main__":
    main()

