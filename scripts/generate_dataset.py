from typing import List
from environments.environment_abstract import Environment, State
from utils import env_utils
from argparse import ArgumentParser
import pickle
import os
import time
import math

from multiprocessing import Queue, Process


def generate_and_save_states(env: Environment, num_states: int, back_max: int, filepath_queue):
    while True:
        filepath = filepath_queue.get()
        if filepath is None:
            break

        start_time = time.time()

        print("Generating data for %s" % filepath)

        # get number of states per backward steps
        assert num_states >= back_max
        states: List[State] = []
        num_back_steps: List[int] = []

        back_step_nums: List[int] = list(range(0, back_max + 1))
        num_states_per_back_step: List[int] = [math.floor(num_states/(back_max + 1)) for _ in back_step_nums]
        num_states_per_back_step[-1] += num_states % (back_max + 1)

        # generate states for each number of backward steps
        for num_states_back_step, back_step in zip(num_states_per_back_step, back_step_nums):
            states_gen: List[State] = env.generate_states(num_states_back_step, (back_step, back_step))

            states.extend(states_gen)
            num_back_steps.extend([back_step]*len(states_gen))

        # save data
        data = dict()
        data['states'] = states
        data['num_back_steps'] = num_back_steps

        pickle.dump(data, open(filepath, "wb"), protocol=-1)

        print("Time for %s: %s" % (filepath, (time.time() - start_time)))


def main():
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument('--env', type=str, required=True, help="Environment")
    parser.add_argument('--back_max', type=int, required=True, help="Maximum number of steps to take backwards from "
                                                                    "goal")
    parser.add_argument('--data_dir', type=str, required=True, help="Directory to save files")
    parser.add_argument('--num_per_file', type=int, default=int(1e6), help="Number of states per file")
    parser.add_argument('--num_files', type=int, default=100, help="Number of files")

    parser.add_argument('--num_procs', type=int, default=1, help="Number of processors to use when generating "
                                                                 "data")

    parser.add_argument('--start_idx', type=int, default=0, help="Start index for file name")

    args = parser.parse_args()

    env: Environment = env_utils.get_environment(args.env)
    assert args.num_per_file >= args.back_max, "Number of states per file should be greater than the or equal to the " \
                                               "number of backwards steps"

    if not os.path.exists(args.data_dir):
        os.mkdir(args.data_dir)

    # make filepath queues
    filepath_queue = Queue()
    filepaths = ["%s/data_%i.pkl" % (args.data_dir, train_idx + args.start_idx) for train_idx in range(args.num_files)]
    for filepath in filepaths:
        filepath_queue.put(filepath)

    # generate_and_save_states(env, args.num_per_file, args.back_max, filepath_queue)

    # start data runners
    data_procs = []
    for _ in range(args.num_procs):
        data_proc = Process(target=generate_and_save_states,
                            args=(env, args.num_per_file, args.back_max, filepath_queue))
        data_proc.daemon = True
        data_proc.start()
        data_procs.append(data_proc)

    # stop data runners
    for _ in range(len(data_procs)):
        filepath_queue.put(None)

    for data_proc in data_procs:
        data_proc.join()


if __name__ == "__main__":
    main()
