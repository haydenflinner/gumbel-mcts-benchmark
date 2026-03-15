import numpy as np
from utils.vectorized_game import VectorizedGame

class VectorizedEnvWrapper:
    """
    A bridge that makes a single 'VectorizedGame' instance look exactly like
    a standard 'BoardGameEnv' for compatibility with mcts_v2.
    """
    def __init__(self, logic, vec_env=None):
        # 1. Initialize the Base Class
        # We pass dummy values because we will override the state management
        # super().__init__(
        #     board_size=logic.BOARD_SHAPE[0], # Assuming square board for Othello/Gomoku
        #     num_stack=1,                     # We handle observation logic manually
        #     has_pass_move=getattr(logic, 'PASS_ACTION', -1) >= 0,
        #     id=f"Vectorized_{logic.GAME_NAME}"
        # )
        
        self.logic = logic
        
        # 2. Setup The Real Engine (VectorizedGame)
        if vec_env is None:
            self.env = VectorizedGame(1, logic)
            self.env.reset()
        else:
            self.env = vec_env
            
        self.game_idx = 0
        
        # 3. Track state locally for the wrapper interface
        self._last_done = False
        self._last_reward = 0.0
        self._winner = None

    # -------------------------------------------------------------------------
    # Overrides: Redirect State Management to VectorizedGame
    # -------------------------------------------------------------------------
    @property
    def winner(self) -> int:
        return self._winner
    
    @property
    def black_player(self) -> int:
        return 1
    
    @property
    def white_player(self) -> int:
        return 0
    
    @property
    def action_dim(self) -> int:
        """Required by Node initialization in uct_search"""
        return self.logic.NUM_ACTIONS
    
    @property
    def has_pass_move(self) -> bool:
        return hasattr(self.logic, 'PASS_ACTION') and self.logic.PASS_ACTION >= 0
    
    @property
    def pass_move(self) -> int:
        return getattr(self.logic, 'PASS_ACTION', None)
    
    @property
    def opponent_player(self):
        """Return the opponent of the current player (used by MCTS for child_to_play)."""
        # For 2-player games with PLAYER_1=1, PLAYER_2=2: opponent = 3 - current
        return 3 - self.to_play

    @property
    def board(self):
        # BoardGameEnv stores self.board. We override the property to read from backend.
        return self.env.boards[self.game_idx]

    @property
    def to_play(self):
        return int(self.env.players[self.game_idx])
    
    @to_play.setter
    def to_play(self, value):
        # MCTS V2 might try to set this manually (rare, but possible in backup/revert)
        self.env.players[self.game_idx] = value

    @property
    def legal_actions(self):
        # Return the float mask (1.0 for legal, 0.0 for illegal)
        return self.env.get_legal_masks()[self.game_idx]

    def reset(self, **kwargs):
        self.env.reset()
        self._last_done = False
        self._last_reward = 0.0
        self._winner = None
        return self.observation()

    def step(self, action: int):
        if self._last_done:
            raise RuntimeError("Game is over, call reset before using step method.")

        # VectorizedGame expects (N,) int32 array
        actions = np.array([action], dtype=np.int32)
        rewards, winners, dones = self.env.step(actions)
        
        self._last_done = bool(dones[self.game_idx])
        self._last_reward = float(rewards[self.game_idx])
        
        if self._last_done:
            self._winner = int(winners[self.game_idx])
            
        # BoardGameEnv step returns: obs, reward, done, info
        return self.observation(), self._last_reward, self._last_done, {}

    def observation(self):
        # Return the raw board state
        flat_board = self.board.flatten().astype(np.float32)  # (42,)
        player = np.array([self.to_play], dtype=np.float32)   # (1,)
        return np.concatenate([flat_board, player]).copy()


    def is_game_over(self) -> bool:
        return self._last_done
    
    @property
    def last_player(self):
        # Used by mcts_v2 assertions.
        # In a strict alternating game, last_player is the one who ISN'T to_play.
        # Note: logic.PLAYER_1 is usually 1, PLAYER_2 is 2.
        p1, p2 = self.logic.PLAYER_1, self.logic.PLAYER_2
        return p2 if self.to_play == p1 else p1

    @property
    def pass_move(self):
        return getattr(self.logic, 'PASS_ACTION', None)

    # -------------------------------------------------------------------------
    # Crucial for MCTS: Deep Copy
    # -------------------------------------------------------------------------
    def __deepcopy__(self, memo):
        # mcts_v2 calls copy.deepcopy(env) for every simulation node!
        
        # 1. Create fresh wrapper
        new_wrapper = VectorizedEnvWrapper(self.logic)
        
        # 2. Copy the underlying VectorizedGame state manually (Fast)
        # We don't want to deepcopy the whole VectorizedGame class, just the arrays for index 0
        new_wrapper.env.boards[0] = self.env.boards[0].copy()
        new_wrapper.env.players[0] = self.env.players[0]
        new_wrapper.env.active_mask[0] = self.env.active_mask[0]
        new_wrapper.env.consecutive_passes[0] = self.env.consecutive_passes[0]
        
        # 3. Copy local wrapper state
        new_wrapper._last_done = self._last_done
        new_wrapper._last_reward = self._last_reward
        new_wrapper._winner = self._winner
        
        return new_wrapper
