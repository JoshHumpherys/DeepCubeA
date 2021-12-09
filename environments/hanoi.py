from typing import List, Tuple, Union
import numpy as np
import torch.nn as nn

from utils.pytorch_models import ResnetModel
from .environment_abstract import Environment, State
from random import randrange


class HanoiState(State):
    __slots__ = ['discs', 'hash']

    def __init__(self, discs: np.ndarray):
        self.discs: np.ndarray = discs
        self.hash = None

    def __hash__(self):
        if self.hash is None:
            self.hash = hash(self.discs.tostring())

        return self.hash

    def __eq__(self, other):
        return np.array_equal(self.discs, other.discs)


class Hanoi(Environment):
    # Moves (swap S, A, T)
    moves: List[str] = ['SA', 'ST', 'AS', 'AT', 'TS', 'TA']
    moves_rev: List[str] = ['AS', 'TS', 'SA', 'TA', 'ST', 'AT']

    def __init__(self, dim: int):
        super().__init__()

        # Validate dim
        MAX_N = 5
        assert dim > 0, 'cant consider hanoi towers with non positive number of towers'
        assert dim < MAX_N, 'cant consider hanoi towers with dim larger than {}'.format(MAX_N)
        self.dim: int = dim
        if self.dim <= 15:
            self.dtype = np.uint8
        else:
            self.dtype = np.int

        # Solved state
        self.goal_discs: np.ndarray = np.concatenate((np.squeeze(np.zeros((self.dim * 2, 1), dtype=self.dtype)), np.arange(1, self.dim + 1, dtype=self.dtype)), axis=0)

    def next_state(self, states: List[HanoiState], action: int) -> Tuple[List[HanoiState], List[float]]:
        """ Get the next state and transition cost given the current state and action

        @param states: List of states
        @param action: Action to take
        @return: Next states, transition costs
        """
        # initialize
        # states_h = np.stack([x.discs for x in states], axis=0)
        states_h = states.copy()
        if isinstance(states_h[0], HanoiState):
            states_h = np.stack([x.discs for x in states], axis=0)
        elif isinstance(states_h[0], np.ndarray):
            states_h = np.array(states_h)

        # get next state
        states_next_h, transition_costs = self._move_h(states_h, action)

        # make states
        # states_next: List[HanoiState] = [HanoiState(x) for x in list(states_next_h)]
        states_next = states_next_h

        return states_next, transition_costs

    def prev_state(self, states: List[HanoiState], action: int) -> List[HanoiState]:
        """ Get the previous state based on the current state and action

        @param states: List of states
        @param action: Action to take to get the previous state
        @return: Previous states
        """
        move: str = self.moves[action]
        move_rev_idx: int = np.where(np.array(self.moves_rev) == np.array(move))[0][0]

        return self.next_state(states, move_rev_idx)[0]

    def generate_goal_states(self, num_states: int, np_format: bool = False) -> Union[List[HanoiState], np.ndarray]:
        """ Generate goal states

                @param num_states: Number of states to generate
                @return: List of states
                """
        if np_format:
            goal_np: np.ndarray = np.expand_dims(self.goal_discs.copy(), 0)
            solved_states: np.ndarray = np.repeat(goal_np, num_states, axis=0)
        else:
            solved_states = List[HanoiState] = [HanoiState(self.goal_discs.copy()) for _ in range(num_states)]

        return solved_states

    def is_solved(self, states: List[HanoiState]) -> np.ndarray:
        """ Returns whether or not state is solved

                @param states: List of states
                @return: Boolean numpy array where the element at index i corresponds to whether or not the
                state at index i is solved
                """
        if isinstance(states[0], HanoiState):
            states_h = np.stack([x.discs for x in states], axis=0)
        elif isinstance(states[0], np.ndarray):
            states_h = np.array(states)

        is_equal = np.equal(states_h, np.expand_dims(self.goal_discs, 0))

        return np.all(is_equal, axis=1)

    def state_to_nnet_input(self, states: List[HanoiState]) -> List[np.ndarray]:
        """ State to numpy arrays to be fed to the neural network

        @param states: List of states
        @return: List of numpy arrays. Each index along the first dimension of each array corresponds to the index of
        a state.
        """
        states_np = np.stack([x.discs for x in states], axis=0)

        representation = [states_np.astype(self.dtype)]

        return representation

    def get_num_moves(self) -> int:
        """ Used for environments with fixed actions. Corresponds to the numbers of each action

        @return: List of action ints
        """
        return len(self.moves)

    def get_nnet_model(self) -> nn.Module:
        """ Get the neural network model for the environment

        @return: neural network model
        """
        # TODO Check with team
        state_dim: int = self.dim * 3
        nnet = ResnetModel(state_dim, self.dim ** 2, 5000, 1000, 4, 1, True)

        return nnet

    def generate_states(self, num_states: int, backwards_range: Tuple[int, int]) -> Tuple[List[HanoiState], List[int]]:

        assert (num_states > 0)
        assert (backwards_range[0] >= 0)
        assert self.fixed_actions, "Environments without fixed actions must implement their own method"

        # Initialize
        scrambs: List[int] = list(range(backwards_range[0], backwards_range[1] + 1))
        num_env_moves: int = self.get_num_moves()

        # Get goal states
        states: np.ndarray = self.generate_goal_states(num_states, np_format=True)

        # Scrambles
        scramble_nums: np.array = np.random.choice(scrambs, num_states)
        num_back_moves: np.array = np.zeros(num_states)

        # Go backward from goal state
        while np.max(num_back_moves < scramble_nums):
            idxs: np.ndarray = np.where((num_back_moves < scramble_nums))[0]
            subset_size: int = int(max(len(idxs) / num_env_moves, 1))
            idxs: np.ndarray = np.random.choice(idxs, subset_size)

            move: int = randrange(num_env_moves)
            # states_to_move = [states[i] for i in idxs]
            states_to_move = states[idxs]
            states_moved = self.prev_state(states_to_move, move)

            for state_moved_idx, state_moved in enumerate(states_moved):
                states[idxs[state_moved_idx]] = state_moved

            num_back_moves[idxs] = num_back_moves[idxs] + 1

        states: List[HanoiState] = [HanoiState(x) for x in list(states)]

        return states, scramble_nums.tolist()

    def expand(self, states: List[State]) -> Tuple[List[List[State]], List[np.ndarray]]:
        """ Generate all children for the state

        @param states: List of states
        @return: Children of each state, Transition costs for each state
        """
        assert self.fixed_actions, "Environments without fixed actions must implement their own method"

        # initialize
        num_states: int = len(states)
        num_env_moves: int = self.get_num_moves()

        states_exp: List[List[State]] = []
        for _ in range(len(states)):
            states_exp.append([])

        tc: np.ndarray = np.empty([num_states, num_env_moves])

        # numpy states
        states_np: np.ndarray = np.stack([state.discs for state in states])

        # for each move, get next states, transition costs, and if solved
        move_idx: int
        move: int
        for move_idx in range(num_env_moves):
            # next state
            states_next_np: np.ndarray
            tc_move: List[float]
            states_next_np, tc_move = self._move_h(states_np, move_idx)

            # transition cost
            tc[:, move_idx] = np.array(tc_move)

            for idx in range(len(states)):
                states_exp[idx].append(HanoiState(states_next_np[idx]))

        # make lists
        tc_l: List[np.ndarray] = [tc[i] for i in range(num_states)]

        return states_exp, tc_l

    def _move_h(self, states_h: np.ndarray, action: int) -> Tuple[np.ndarray, List[float]]:
        states_next_h: np.ndarray = states_h.copy()

        # Establish IDX constants
        pole_s_start_idx = 0
        pole_s_stop_idx = self.dim
        pole_a_start_idx = self.dim
        pole_a_stop_idx = self.dim * 2
        pole_t_start_idx = (self.dim * 2)
        pole_t_stop_idx = self.dim * 3

        ##### Perform action on each states_h
        # Get string of move
        move: str = self.moves[action]

        # Get starting pole and ending pole
        pole_start_id = move[0]
        pole_end_id = move[1]

        # Get top disk on staring pole and ending pole
        if pole_start_id == 'S':
            pole_start = states_h[:, pole_s_start_idx:pole_s_stop_idx]
        elif pole_start_id == 'A':
            pole_start = states_h[:, pole_a_start_idx:pole_a_stop_idx]
        elif pole_start_id == 'T':
            pole_start = states_h[:, pole_t_start_idx:pole_t_stop_idx]

        # pole_start_empty_idx = np.where(~pole_start.any(axis=1))[0]
        # Find first positive disk, else set disk value to 0
        pole_start_top_disk_idx = np.where(pole_start > 0, pole_start, np.inf).argmin(axis=1)
        pole_start_top_disk = pole_start[range(pole_start.shape[0]), pole_start_top_disk_idx]
        # pole_start_top_disk = np.amin(pole_start, axis=1)
        # pole_start_top_disk_idx = pole_start_positive_disk_idx  # np.argmin(pole_start, axis=1)

        if pole_end_id == 'S':
            pole_end = states_h[:, pole_s_start_idx:pole_s_stop_idx]
        elif pole_end_id == 'A':
            pole_end = states_h[:, pole_a_start_idx:pole_a_stop_idx]
        elif pole_end_id == 'T':
            pole_end = states_h[:, pole_t_start_idx:pole_t_stop_idx]

        pole_end_top_disk_idx = np.where(pole_end > 0, pole_end, np.inf).argmin(axis=1)
        pole_end_top_disk = pole_end[range(pole_end.shape[0]), pole_end_top_disk_idx]
        # pole_end_top_disk = np.amin(pole_end, axis=1)
        # pole_end_top_disk_idx = np.argmin(pole_end, axis=1)

        mask_legal_move = np.logical_and(pole_start_top_disk != 0, np.logical_or(pole_end_top_disk > pole_start_top_disk, np.zeros(pole_end_top_disk.shape) == pole_end_top_disk))
        if np.all(np.invert(mask_legal_move)):
            states_next_h = states_h
        else:

            # Create updated pole_start
            pole_start_updated = pole_start.copy()
            pole_start_updated[mask_legal_move, pole_start_top_disk_idx[mask_legal_move]] = 0
            # If not a legal move, no update to pole_start

            # Create updated pole_end
            pole_end_updated = pole_end.copy()
            # Handle empty pole_end
            pole_end_empty_idx = np.where(~pole_end.any(axis=1))[0]
            pole_end_updated[pole_end_empty_idx, self.dim - 1] = pole_start_top_disk[pole_end_empty_idx]
            # Handle nonempty pole_end
            mask_pole_end_nonempty = np.ones(pole_end.shape[0], np.bool)
            mask_pole_end_nonempty[pole_end_empty_idx] = 0
            if np.any(mask_pole_end_nonempty):
                pole_end_updated[mask_pole_end_nonempty, pole_end_top_disk_idx[mask_pole_end_nonempty] - 1] = pole_start_top_disk[mask_pole_end_nonempty]
            # Handle illegal moves
            pole_end_updated[np.invert(mask_legal_move), :] = pole_end[np.invert(mask_legal_move), :]

            # Update states_h
            if pole_start_id == 'S':
                states_next_h[:, pole_s_start_idx:pole_s_stop_idx] = pole_start_updated
            elif pole_start_id == 'A':
                states_next_h[:, pole_a_start_idx:pole_a_stop_idx] = pole_start_updated
            elif pole_start_id == 'T':
                states_next_h[:, pole_t_start_idx:pole_t_stop_idx] = pole_start_updated

            if pole_end_id == 'S':
                states_next_h[:, pole_s_start_idx:pole_s_stop_idx] = pole_end_updated
            elif pole_end_id == 'A':
                states_next_h[:, pole_a_start_idx:pole_a_stop_idx] = pole_end_updated
            elif pole_end_id == 'T':
                states_next_h[:, pole_t_start_idx:pole_t_stop_idx] = pole_end_updated

        # transition costs
        transition_costs: List[float] = [1.0 for _ in range(states_h.shape[0])]

        return states_next_h, transition_costs


