import numpy as np
from numba import njit


def _is_numba_dispatcher(fn) -> bool:
    try:
        from numba.core.dispatcher import Dispatcher
        return isinstance(fn, Dispatcher)
    except ImportError:
        return False

def make_batch_legal_mask_kernel(get_valid_mask_func, num_actions):
    @njit
    def _batch_mask_kernel(boards, players, active_mask):
        n = len(boards)
        # Allocate output: (N, NUM_ACTIONS)
        legal_masks = np.zeros((n, num_actions), dtype=np.float32)
        
        for i in range(n):
            if active_mask[i]:
                # Call the game-specific logic
                mask = get_valid_mask_func(boards[i], players[i])
                # Convert bool/int mask to float for the neural network
                for a in range(num_actions):
                    if mask[a]:
                        legal_masks[i, a] = 1.0
        return legal_masks
    return _batch_mask_kernel


def make_batch_legal_mask_python(get_valid_mask_func, num_actions):
    def _batch_mask_python(boards, players, active_mask):
        n = len(boards)
        legal_masks = np.zeros((n, num_actions), dtype=np.float32)
        for i in range(n):
            if active_mask[i]:
                mask = get_valid_mask_func(boards[i], players[i])
                legal_masks[i, :] = np.asarray(mask, dtype=np.float32)
        return legal_masks
    return _batch_mask_python

def make_batch_step_kernel(fast_step_func, pass_action_val):
    """
    Factory that creates a game-specific Numba kernel.
    pass_action_val: The integer action ID for 'Pass' (or -1 if none).
    """
    @njit
    def _batch_step_kernel(boards, players, moves, active_mask, consecutive_passes):
        n = len(boards)
        rewards = np.zeros(n, dtype=np.float32)
        winners = np.zeros(n, dtype=np.int8)
        dones = np.zeros(n, dtype=np.bool_)
        
        for i in range(n):
            if active_mask[i]:
                # fast_step_func is "baked in" to this specific kernel
                r, w, d, next_b = fast_step_func(boards[i], moves[i], players[i])

                boards[i] = next_b

                # 3. Handle Double Pass Termination (Othello Specific)
                if pass_action_val >= 0:
                    if moves[i] == pass_action_val:
                        consecutive_passes[i] += 1
                    else:
                        consecutive_passes[i] = 0
                        
                    if consecutive_passes[i] >= 2:
                        d = True
                        # Recalculate Winner based on disc count
                        # (Since fast_step didn't know it was game over, it returned 0/False)
                        
                        # Count pieces (1=P1, 2=P2)
                        p1_count = 0
                        p2_count = 0
                        # Flat iteration for speed
                        flat_board = boards[i].ravel()
                        for k in range(flat_board.size):
                            if flat_board[k] == 1: p1_count += 1
                            elif flat_board[k] == 2: p2_count += 1
                        
                        if p1_count > p2_count: w = 1
                        elif p2_count > p1_count: w = 2
                        else: w = 0 # Draw
                        
                        # Recalculate reward relative to current player
                        if w == players[i]: r = 1.0
                        elif w == 0: r = 0.0
                        else: r = -1.0

                # 4. Write outputs
                rewards[i] = r
                winners[i] = w
                dones[i] = d     

                # Auto-switch player if game not done
                # Note: This assumes 2-player alternating games (1->2->1)
                # If you have games with different turns (e.g. skip turn), 
                # fast_step should return the next player instead.
                # if not d:
                players[i] = 3 - players[i] 
                    
        return rewards, winners, dones
        
    return _batch_step_kernel


def make_batch_step_python(fast_step_func, pass_action_val):
    def _batch_step_python(boards, players, moves, active_mask, consecutive_passes):
        n = len(boards)
        rewards = np.zeros(n, dtype=np.float32)
        winners = np.zeros(n, dtype=np.int8)
        dones = np.zeros(n, dtype=np.bool_)

        for i in range(n):
            if not active_mask[i]:
                continue

            r, w, d, next_b = fast_step_func(boards[i], int(moves[i]), int(players[i]))
            boards[i] = next_b

            if pass_action_val >= 0:
                if moves[i] == pass_action_val:
                    consecutive_passes[i] += 1
                else:
                    consecutive_passes[i] = 0

                if consecutive_passes[i] >= 2:
                    d = True
                    p1_count = 0
                    p2_count = 0
                    flat_board = boards[i].ravel()
                    for k in range(flat_board.size):
                        if flat_board[k] == 1:
                            p1_count += 1
                        elif flat_board[k] == 2:
                            p2_count += 1

                    if p1_count > p2_count:
                        w = 1
                    elif p2_count > p1_count:
                        w = 2
                    else:
                        w = 0

                    if w == players[i]:
                        r = 1.0
                    elif w == 0:
                        r = 0.0
                    else:
                        r = -1.0

            rewards[i] = r
            winners[i] = w
            dones[i] = d
            players[i] = 3 - players[i]

        return rewards, winners, dones

    return _batch_step_python

class VectorizedGame:
    def __init__(self, n_games, game_logic):
        """
        Args:
            n_games: Number of parallel games
            game_logic: The loaded logic module (containing BOARD_SHAPE, fast_step)
        """
        self.n_games = n_games
        self.logic = game_logic

        self.use_history = getattr(self.logic, 'USE_HISTORY', False)
        self.history_steps = getattr(self.logic, 'HISTORY_STEPS', 0)

        self.winners = np.zeros(n_games, dtype=np.int8)
        
        # 1. Dynamic Shape
        self.boards = np.zeros((n_games, *self.logic.BOARD_SHAPE), dtype=np.int8)
        self.players = np.ones(n_games, dtype=np.int8) # Player 1 starts
        self.active_mask = np.ones(n_games, dtype=np.bool_)
        self.consecutive_passes = np.zeros(n_games, dtype=np.int8)
        self.ply_counts = np.zeros(n_games, dtype=np.int32)
        self.position_hash_fn = getattr(self.logic, 'position_hash', None)
        self.position_history = [dict() for _ in range(n_games)] if callable(self.position_hash_fn) else None

        if self.use_history:
            assert self.logic.GAME_NAME == 'chess', "History tracking is currently only implemented for chess"
            # We only store the 64 piece squares in history, not the meta bits
            self.history_buffer = np.zeros((n_games, self.history_steps, 64), dtype=np.int8)

        pass_action = getattr(self.logic, 'PASS_ACTION', -1)
        self._use_numba_kernels = (
            _is_numba_dispatcher(self.logic.fast_step)
            and _is_numba_dispatcher(self.logic.get_valid_mask)
        )

        # 2. Dynamic kernel selection
        if self._use_numba_kernels:
            self.step_kernel = make_batch_step_kernel(self.logic.fast_step, pass_action)
            self.legal_mask_kernel = make_batch_legal_mask_kernel(
                self.logic.get_valid_mask,
                self.logic.NUM_ACTIONS
            )
        else:
            self.step_kernel = make_batch_step_python(self.logic.fast_step, pass_action)
            self.legal_mask_kernel = make_batch_legal_mask_python(
                self.logic.get_valid_mask,
                self.logic.NUM_ACTIONS
            )

        self.reset() # othello needs this
    
    def reset(self):
        initial_board = self.logic.get_initial_board()
        
        for i in range(self.n_games):
            self.boards[i] = initial_board

            if self.use_history:
                # Fill history with the initial piece configuration
                # Assuming pieces are the first 64 indices of the board array
                for step in range(self.history_steps):
                    self.history_buffer[i, step] = initial_board[:64]

        self.players.fill(self.logic.PLAYER_1)
        self.active_mask.fill(True)
        self.consecutive_passes.fill(0)
        self.ply_counts.fill(0)
        self.winners.fill(0)

        if self.position_history is not None:
            for i in range(self.n_games):
                self.position_history[i].clear()
                key = self.position_hash_fn(self.boards[i])
                self.position_history[i][key] = 1
        
    def step(self, moves, active_indices=None):
        """
        moves: (N,) array of ints
        """
        # Call the dynamically generated kernel
        active_before = self.active_mask.copy()

        rewards, winners, dones = self.step_kernel(
            self.boards, 
            self.players, 
            moves.astype(np.int32), 
            self.active_mask,
            self.consecutive_passes
        )

        if self.use_history:
            just_stepped = active_before & (~dones)
            if np.any(just_stepped):

                self.history_buffer[just_stepped] = np.roll(
                    self.history_buffer[just_stepped], shift=-1, axis=1
                )

                self.history_buffer[just_stepped, -1] = self.boards[just_stepped, :64]

        if np.any(active_before):
            self.ply_counts[active_before] += 1

        max_moves = getattr(self.logic, 'MAX_MOVES', None)
        if max_moves is not None:
            maxed = active_before & (~dones) & (self.ply_counts >= int(max_moves))
            if np.any(maxed):
                dones[maxed] = True
                winners[maxed] = 0
                rewards[maxed] = 0.0

        if self.position_history is not None:
            for i in range(self.n_games):
                if not self.active_mask[i] or dones[i]:
                    continue

                key = self.position_hash_fn(self.boards[i])
                count = self.position_history[i].get(key, 0) + 1
                self.position_history[i][key] = count

                if count >= 3:
                    dones[i] = True
                    winners[i] = 0
                    rewards[i] = 0.0
        
        # Update active mask based on dones
        just_finished = self.active_mask & dones
        self.active_mask[just_finished] = False
        self.winners = np.where(just_finished, winners, self.winners)
        return rewards, winners, dones

    def get_legal_masks(self, indices=None):
        """
        Returns (N, NUM_ACTIONS) float array where 1.0 is legal, 0.0 is illegal.
        If indices is provided, only computes for those games (others are 0).
        """
        # If indices provided, we need a temporary mask, or just pass full active_mask
        # For simplicity in self-play, we usually just want masks for currently active games.
        
        return self.legal_mask_kernel(self.boards, self.players, self.active_mask)
    
    @property
    def any_active(self):
        return np.any(self.active_mask)
    
    def get_observation(self):
        """
        Returns the data actually fed to the Neural Network.
        For Chess: (N, 515)
        """
        if not self.use_history:
            return self.boards

        obs_history = self.history_buffer.reshape(self.n_games, -1)        
        meta_bits = self.boards[:, 64:67]
        return np.concatenate([obs_history, meta_bits], axis=1).astype(np.float32)
    
    def get_search_state(self):
        """
        Returns the FULL memory block required by the MCTS node.
        For Chess: [515 observation] + [69 raw state] = 584 bytes.
        """
        obs = self.get_observation()
        if not self.use_history:
            return obs.astype(np.int8)
            
        obs_int8 = np.round(obs).astype(np.int8)

        # Concatenate Observation + Raw Shadow for MCTS storage compatibility
        return np.concatenate([obs_int8, self.boards], axis=1)
    
    def get_scores(
        self,
        player_perspective: np.ndarray,
        results: np.ndarray,
        game_lengths: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Get scores for each game from the given player's perspective.
        
        Args:
            player_perspective: (n_games,) array of player IDs to score from
            results: (n_games,) game results (+1 win, 0 draw, -1 loss)
            game_lengths: optional (n_games,) number of plies played in each game
            
        Returns:
            scores: (n_games,) array of scores (positive = good for player)
        """
        if game_lengths is None:
            return self.logic.get_scores(self.boards, player_perspective, results)
        return self.logic.get_scores(self.boards, player_perspective, results, game_lengths)