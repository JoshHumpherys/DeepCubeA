from typing import List, Tuple, Union
import numpy as np
import torch.nn as nn

from utils.env_utils import create_nnet_with_overridden_params
from .environment_abstract import Environment, State


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
        MAX_N = 12
        assert dim > 0, 'cant consider hanoi towers with non positive number of towers'
        assert dim < MAX_N, 'cant consider hanoi towers with dim larger than {}'.format(MAX_N)
        self.dim: int = dim
        if self.dim <= 15:
            self.dtype = np.uint8
        else:
            self.dtype = np.int

        # Solved state
        self.goal_discs: np.ndarray = np.concatenate((np.squeeze(np.zeros((self.dim * 2, 1), dtype=self.dtype)), np.arange(1, self.dim + 1, dtype=self.dtype)), axis=0)

        # ND Array Idx
        self.pole_s_start_idx = 0
        self.pole_s_stop_idx = self.dim
        self.pole_a_start_idx = self.dim
        self.pole_a_stop_idx = self.dim * 2
        self.pole_t_start_idx = (self.dim * 2)
        self.pole_t_stop_idx = self.dim * 3

    def next_state(self, states: List[HanoiState], action: np.ndarray) -> Tuple[List[HanoiState], List[float]]:
        """ Get the next state and transition cost given the current state and action

        @param states: List of states
        @param action: Action to take
        @return: Next states, transition costs
        """
        # initialize
        states_h = states.copy()
        if isinstance(states_h[0], HanoiState):
            states_h = np.stack([x.discs for x in states], axis=0)
        elif isinstance(states_h[0], np.ndarray):
            states_h = np.array(states_h)

        # get next state
        states_next_h, transition_costs = self._move_h(states_h, action)

        # make states
        states_next = states_next_h

        return states_next, transition_costs

    def prev_state(self, states: List[HanoiState], action: np.ndarray) -> List[HanoiState]:
        """ Get the previous state based on the current state and action

        @param states: List of states
        @param action: Action to take to get the previous state
        @return: Previous states
        """

        return self.next_state(states, action)[0]

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
        kwargs = dict(state_dim=self.dim * 3, one_hot_depth=self.dim ** 2, h1_dim=5000, resnet_dim=1000,
                      num_resnet_blocks=4, out_dim=1, batch_norm=True)

        return create_nnet_with_overridden_params(kwargs)

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

            # Get states_to_move, which is a subset of the state ndarray made up of the specific rows
            # with rows defined by idxs
            states_to_move = states[idxs]

            # Get move, which is a list of move numbers, each idx corresponding to the random move
            # that will be applied to the row in states_to_move
            # NOTE: the random move can only be a legal move
            moves_possible_for_state = self._get_valid_moves(states_to_move)
            # Choose random moves
            moves_for_state = np.expand_dims(np.array([np.random.choice(np.squeeze(x)) for x in moves_possible_for_state]), 1)
            # Get previous state for states_to_move given move
            states_moved = self.prev_state(states_to_move, moves_for_state)

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

        states_exp: List[List[State]] = []
        tc: List[List[State]] = []
        for _ in range(len(states)):
            states_exp.append([])
            tc.append([])

        # tc: np.ndarray = np.empty([num_states, num_env_moves])

        # numpy states
        states_h: np.ndarray = np.stack([state.discs for state in states])

        # determine which moves are possible for state
        moves_possible_for_state = self._get_valid_moves(states_h)

        for idx in range(len(states)):
            state_h = np.expand_dims(states_h[idx, :], 0)

            for move_idx in range(len(moves_possible_for_state[idx])):
                move = np.expand_dims(moves_possible_for_state[idx][move_idx], 1)
                state_next_h, tc_move = self._move_h(state_h, move)

                states_exp[idx].append(HanoiState(state_next_h))
                tc[idx].append(tc_move)

        tc_l = tc

        return states_exp, tc_l

    def _get_valid_moves(self, states_h: np.ndarray):

        # Break up ndarray into separate poles
        pole_src, pole_aux, pole_tgt = self._break_states_h_into_poles(states_h)

        ### Get Top Disk and Top Disk Idx for each pole
        # SOURCE
        pole_src_top_disk, _ = self._get_top_disk_and_top_disk_idx(pole_src)

        # AUXILLARY DISK
        pole_aux_top_disk, _ = self._get_top_disk_and_top_disk_idx(pole_aux)

        # TARGET
        pole_tgt_top_disk, _ = self._get_top_disk_and_top_disk_idx(pole_tgt)

        ### Check if legal move
        # SOURCE --> AUXILLARY
        mask_SA = self._is_valid_move(pole_src_top_disk, pole_aux_top_disk)

        # SOURCE --> TARGET
        mask_ST = self._is_valid_move(pole_src_top_disk, pole_tgt_top_disk)

        # AUXILLARY --> SOURCE
        mask_AS = self._is_valid_move(pole_aux_top_disk, pole_src_top_disk)

        # AUXILLARY --> TARGET
        mask_AT = self._is_valid_move(pole_aux_top_disk, pole_tgt_top_disk)

        # TARGET --> SOURCE
        mask_TS = self._is_valid_move(pole_tgt_top_disk, pole_src_top_disk)

        # TARGET --> AUXILLARY
        mask_TA = self._is_valid_move(pole_tgt_top_disk, pole_aux_top_disk)

        ### Combine masks into array
        # Column order must follow order to moves at the top of the class
        # moves: List[str] = ['SA', 'ST', 'AS', 'AT', 'TS', 'TA']
        mask_legal_moves = np.concatenate((mask_SA, mask_ST, mask_AS, mask_AT, mask_TS, mask_TA), axis=1)

        ### Get valid move idx
        mask_legal_moves_idx = [np.argwhere(x > 0) for x in mask_legal_moves]

        return mask_legal_moves_idx

    def _break_states_h_into_poles(self, states_h: np.ndarray):
        # Break up ndarray into separate poles
        pole_src = states_h[:, self.pole_s_start_idx:self.pole_s_stop_idx]
        pole_aux = states_h[:, self.pole_a_start_idx:self.pole_a_stop_idx]
        pole_tgt = states_h[:, self.pole_t_start_idx:self.pole_t_stop_idx]

        return pole_src, pole_aux, pole_tgt

    def _get_top_disk_and_top_disk_idx(self, pole):
        top_disk_idx = np.where(pole > 0, pole, np.inf).argmin(axis=1)
        top_disk = pole[range(pole.shape[0]), top_disk_idx]

        return top_disk, top_disk_idx

    def _is_valid_move(self, pole_start_top_disk, pole_end_top_disk):
        # Move is legal if:
        # 1) a disk is present
        # 2) disk number from start is less than disc number from end OR end pole is empty (disk value = 0)

        mask_rule_1 = pole_start_top_disk != 0
        mask_rule_2 = np.logical_or(pole_end_top_disk > pole_start_top_disk,
                                    np.zeros(pole_end_top_disk.shape) == pole_end_top_disk)
        mask_legal_move = np.logical_and(mask_rule_1, mask_rule_2)

        return np.expand_dims(mask_legal_move, 1)

    def _get_positive_idx(self, mask):
        return [np.argwhere(mask > 0)]

    def _move_disk(self, pole_start, pole_end):
        # Initialize
        pole_start_updated = pole_start.copy()
        pole_end_updated = pole_end.copy()

        # Get top disk and idx of top disk
        pole_start_top_disk, pole_start_top_disk_idx = self._get_top_disk_and_top_disk_idx(pole_start)
        pole_end_top_disk, pole_end_top_disk_idx = self._get_top_disk_and_top_disk_idx(pole_end)

        # Update starting pole by replacing top disk with 0
        pole_start_updated[0, pole_start_top_disk_idx] = 0

        # Update ending pole
        if pole_end_top_disk == 0:
            # If ending pole is empty
            pole_end_updated[0, self.dim - 1] = pole_start[0, pole_start_top_disk_idx]
        else:
            pole_end_updated[0, pole_end_top_disk_idx - 1] = pole_start[0, pole_start_top_disk_idx]

        return pole_start_updated, pole_end_updated

    def _move_h(self, states_h: np.ndarray, action: int) -> Tuple[np.ndarray, List[float]]:
        states_next_h: np.ndarray = states_h.copy()

        # Get individual poles
        pole_src, pole_aux, pole_tgt = self._break_states_h_into_poles(states_next_h)

        for i in range(states_h.shape[0]):
            src = np.expand_dims(pole_src[i, :], 0)
            aux = np.expand_dims(pole_aux[i, :], 0)
            tgt = np.expand_dims(pole_tgt[i, :], 0)
            move = self.moves[action[i][0]]

            ########## Process disk move
            ### Initialize
            src_updated = src.copy()
            aux_updated = aux.copy()
            tgt_updated = tgt.copy()

            ### Get IDs of start and end poles
            pole_start_id = move[0]
            pole_end_id = move[1]

            ### Move disk from start pole to end pole
            if pole_start_id == 'S' and pole_end_id == 'A':
                src_updated, aux_updated = self._move_disk(src, aux)
            elif pole_start_id == 'S' and pole_end_id == 'T':
                src_updated, tgt_updated = self._move_disk(src, tgt)
            elif pole_start_id == 'A' and pole_end_id == 'S':
                aux_updated, src_updated = self._move_disk(aux, src)
            elif pole_start_id == 'A' and pole_end_id == 'T':
                aux_updated, tgt_updated = self._move_disk(aux, tgt)
            elif pole_start_id == 'T' and pole_end_id == 'S':
                tgt_updated, src_updated = self._move_disk(tgt, src)
            elif pole_start_id == 'T' and pole_end_id == 'A':
                tgt_updated, aux_updated = self._move_disk(tgt, aux)

            # Update pole
            pole_src[i, :] = src_updated
            pole_aux[i, :] = aux_updated
            pole_tgt[i, :] = tgt_updated

        states_next_h = np.concatenate((pole_src, pole_aux, pole_tgt), axis=1)

        # transition costs
        transition_costs: List[float] = [1.0 for _ in range(states_h.shape[0])]

        return states_next_h, transition_costs
