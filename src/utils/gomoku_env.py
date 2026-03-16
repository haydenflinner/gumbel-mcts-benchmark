"""
Gomoku (Five in a Row) Environment compatible with mcts_v2 (reference.py).

Standalone env that returns flat observations: [flat_board (225), current_player (1)].
Uses numba-accelerated kernels from game_logic.gomoku.
"""

import numpy as np
from typing import Tuple
from game_logic.gomoku import GomokuLogic, fast_step, get_valid_mask, BOARD_SIZE, NUM_ACTIONS


class GomokuEnv:
    """
    Gomoku environment with flat-observation interface for mcts_v2 compatibility.

    Board is 15x15. Actions are flat indices 0-224.
    Observation is (226,): [flat board (225), current player (1)].
    """

    def __init__(self):
        self.logic = GomokuLogic()
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        self.to_play = 1
        self.steps = 0
        self._winner = None
        self._done = False
        self._last_player = None
        self._legal_actions = None

    @property
    def action_dim(self) -> int:
        return NUM_ACTIONS

    @property
    def winner(self):
        return self._winner

    @property
    def opponent_player(self) -> int:
        return 3 - self.to_play

    @property
    def last_player(self):
        return self._last_player

    @property
    def legal_actions(self) -> np.ndarray:
        if self._legal_actions is None:
            self._legal_actions = get_valid_mask(self.board, self.to_play)
        return self._legal_actions

    def reset(self) -> np.ndarray:
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        self.to_play = 1
        self.steps = 0
        self._winner = None
        self._done = False
        self._last_player = None
        self._legal_actions = None
        return self.observation()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        if self._done:
            raise RuntimeError("Game is over, call reset.")

        if self.legal_actions[action] < 1.0:
            raise ValueError(f"Illegal action {action}")

        new_board = self.board.copy()
        reward, winner, done, new_board = fast_step(new_board, action, self.to_play)

        self._last_player = self.to_play
        self.board = new_board
        self.to_play = 3 - self.to_play
        self.steps += 1
        self._done = done
        self._legal_actions = None

        if done:
            self._winner = int(winner)
            r = 1.0 if winner == self._last_player else (0.0 if winner == 0 else -1.0)
            return self.observation(), r, True, {}

        return self.observation(), 0.0, False, {}

    def observation(self) -> np.ndarray:
        flat_board = self.board.flatten().astype(np.float32)  # (225,)
        player = np.array([self.to_play], dtype=np.float32)   # (1,)
        return np.concatenate([flat_board, player])            # (226,)

    def is_game_over(self) -> bool:
        return self._done

    def render(self) -> str:
        symbols = {0: ".", 1: "X", 2: "O"}
        s = f"\n  Gomoku - {'Black' if self.to_play == 1 else 'White'} to play  (step {self.steps})\n"
        s += "   " + " ".join(f"{i:2d}" for i in range(BOARD_SIZE)) + "\n"
        for r in range(BOARD_SIZE):
            s += f"{r:2d} "
            for c in range(BOARD_SIZE):
                s += f" {symbols[self.board[r, c]]} "
            s += "\n"
        if self._winner is not None:
            s += f"Game Over. Winner: {self._winner}\n"
        return s

    def __deepcopy__(self, memo):
        new = GomokuEnv()
        new.board = self.board.copy()
        new.to_play = self.to_play
        new.steps = self.steps
        new._winner = self._winner
        new._done = self._done
        new._last_player = self._last_player
        new._legal_actions = None
        return new


if __name__ == "__main__":
    env = GomokuEnv()
    obs = env.reset()
    print(f"Observation shape: {obs.shape}")  # (226,)
    print(env.render())

    while not env.is_game_over():
        legals = np.where(env.legal_actions >= 1.0)[0]
        if len(legals) == 0:
            break
        a = np.random.choice(legals)
        obs, r, done, _ = env.step(a)
        if done:
            print(env.render())
            print(f"Result: {r}")