"""
Utility to generate interesting test positions for any board game.
Used for testing MCTS implementations across different games.
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class TestPosition:
    """A test position with metadata."""
    move_sequence: List[int]
    description: str
    expected_game_over: bool = False
    is_tactical: bool = False
    # Optional: provide board + player directly (bypasses move_sequence replay)
    board: Optional[np.ndarray] = None
    player: Optional[int] = None


def get_legal_moves(logic, board: np.ndarray, player: int) -> List[int]:
    """Get list of legal move indices for a position."""
    mask = logic.get_valid_mask(board, player)
    return [i for i in range(logic.NUM_ACTIONS) if mask[i]]


def play_sequence(logic, moves: List[int]) -> Tuple[np.ndarray, int, bool, Optional[int]]:
    """
    Play a sequence of moves and return the resulting state.
    
    Returns:
        (board, current_player, is_game_over, winner)
    """
    board = logic.get_initial_board().copy()
    player = logic.PLAYER_1
    
    for move in moves:
        reward, winner, done, board = logic.fast_step(board, move, player)
        if done:
            return board, player, True, winner
        player = 3 - player  # Switch player
    
    return board, player, False, None


def resolve_position(logic, pos: TestPosition) -> Tuple[np.ndarray, int, bool, Optional[int]]:
    """
    Resolve a TestPosition to (board, player, done, winner).
    Uses board/player directly if provided, otherwise replays move_sequence.
    """
    if pos.board is not None and pos.player is not None:
        return pos.board.copy(), pos.player, pos.expected_game_over, None
    return play_sequence(logic, pos.move_sequence)


def generate_blocking_threats(logic) -> List[TestPosition]:
    """
    Attempts to find positions where the current player MUST block an 
    immediate win by the opponent.
    """
    tactical_positions = []
    game_name = getattr(logic, 'GAME_NAME', '').lower()
    
    # We only run this for games that support 'get_winning_moves' 
    if not hasattr(logic, 'get_winning_moves'):
        return []

    # Attempt to find threats through random play
    for seed in range(50, 100):
        np.random.seed(seed)
        board = logic.get_initial_board().copy()
        player = logic.PLAYER_1
        sequence = []
        
        for _ in range(logic.MAX_MOVES):
            # Check if OPPONENT has a winning move right now
            opponent = 3 - player
            threats = logic.get_winning_moves(board, opponent)
            
            if len(threats) > 0:
                # This is a perfect test position! 
                # Current player MUST play in one of the 'threats' indices.
                tactical_positions.append(TestPosition(
                    move_sequence=sequence.copy(),
                    description=f"Tactical: {game_name} must block opponent win",
                    is_tactical=True
                ))
                # We found one for this seed, move to next seed
                break
                
            # Otherwise, play a random move to keep building the board
            legal = get_legal_moves(logic, board, player)
            if not legal: break
            move = np.random.choice(legal)
            _, _, done, board = logic.fast_step(board, move, player)
            sequence.append(int(move))
            
            if done: break
            player = 3 - player
            
        if len(tactical_positions) >= 5: # Limit to 5 unique tactical positions
            break
            
    return tactical_positions

def generate_test_positions(logic, 
                            num_random_positions: int = 3,
                            max_moves_for_random: int = 20,
                            seed: int = 42) -> List[TestPosition]:
    """
    Generate a list of interesting test positions for any game.
    
    Generates:
    1. Empty board (starting position)
    2. After 1 move
    3. After 2 moves  
    4. A few random mid-game positions
    5. A near-terminal position (if findable)
    
    Args:
        logic: Game logic module with BOARD_SHAPE, fast_step, get_valid_mask, etc.
        num_random_positions: Number of random mid-game positions to generate
        max_moves_for_random: Maximum moves for random positions
        seed: Random seed for reproducibility
        
    Returns:
        List of TestPosition objects
    """
    np.random.seed(seed)
    positions = []
    
    # 1. Empty board
    positions.append(TestPosition(
        move_sequence=[],
        description="Empty board (starting position)"
    ))
    
    # 2-4. First few moves (1, 2, 3 moves)
    board = logic.get_initial_board().copy()
    player = logic.PLAYER_1
    current_sequence = []
    
    for num_moves in [1, 2, 3]:
        legal = get_legal_moves(logic, board, player)
        if not legal:
            break
            
        # Pick a "central" move if possible (often more interesting)
        # For grid games, prefer middle columns/positions
        mid_idx = len(legal) // 2
        move = legal[mid_idx]
        
        reward, winner, done, board = logic.fast_step(board, move, player)
        current_sequence.append(move)
        
        positions.append(TestPosition(
            move_sequence=current_sequence.copy(),
            description=f"After {num_moves} move(s)",
            expected_game_over=done
        ))
        
        if done:
            break
        player = 3 - player
    
    # 5. Random mid-game positions
    for i in range(num_random_positions):
        board = logic.get_initial_board().copy()
        player = logic.PLAYER_1
        sequence = []
        
        # Play random number of moves (5 to max_moves)
        num_moves = np.random.randint(5, max_moves_for_random + 1)
        
        for _ in range(num_moves):
            legal = get_legal_moves(logic, board, player)
            if not legal:
                break
                
            move = np.random.choice(legal)
            reward, winner, done, board = logic.fast_step(board, move, player)
            sequence.append(int(move))
            
            if done:
                break
            player = 3 - player
        
        # Only add if game is not over (we want playable positions)
        if not done and sequence:
            positions.append(TestPosition(
                move_sequence=sequence,
                description=f"Random position ({len(sequence)} moves)",
                expected_game_over=False
            ))
    
    # 6. Try to find a near-terminal position (many pieces, but not game over)
    near_terminal = _find_near_terminal_position(logic, seed=seed+100)
    if near_terminal:
        positions.append(near_terminal)
    
    return positions


def _find_near_terminal_position(logic, 
                                  max_attempts: int = 50,
                                  seed: int = 42) -> Optional[TestPosition]:
    """
    Try to find a position that is close to game end but not over.
    This is useful for testing edge cases.
    """
    np.random.seed(seed)
    
    best_sequence = None
    best_move_count = 0
    
    for _ in range(max_attempts):
        board = logic.get_initial_board().copy()
        player = logic.PLAYER_1
        sequence = []
        
        # Play until game over or stuck
        for _ in range(100):  # Safety limit
            legal = get_legal_moves(logic, board, player)
            if not legal:
                break
                
            move = np.random.choice(legal)
            reward, winner, done, board = logic.fast_step(board, move, player)
            sequence.append(int(move))
            
            if done:
                # Game ended - the position BEFORE this move is near-terminal
                if len(sequence) > best_move_count + 1:
                    best_sequence = sequence[:-1]  # Exclude the winning move
                    best_move_count = len(best_sequence)
                break
            
            player = 3 - player
    
    if best_sequence and len(best_sequence) >= 5:
        return TestPosition(
            move_sequence=best_sequence,
            description=f"Near-terminal position ({len(best_sequence)} moves)",
            expected_game_over=False
        )
    
    return None


def generate_game_specific_positions(logic) -> List[TestPosition]:
    """
    Generate game-specific interesting positions based on game name.
    Falls back to generic positions if game not recognized.
    """
    game_name = getattr(logic, 'GAME_NAME', '').lower()
    
    if 'connect4' in game_name or 'connect_four' in game_name:
        return _connect4_positions(logic)
    elif 'othello' in game_name or 'reversi' in game_name:
        return _othello_positions(logic)
    elif 'gomoku' in game_name:
        return _gomoku_positions(logic)
    elif 'chess' in game_name:
        return _chess_positions(logic)
    else:
        # Generic positions for unknown games
        return generate_test_positions(logic)


def _connect4_positions(logic) -> List[TestPosition]:
    """Connect4-specific interesting positions."""
    positions = generate_test_positions(logic)
    
    # Add Connect4-specific tactical positions
    # Center column control
    positions.append(TestPosition(
        move_sequence=[3, 3, 3, 3],  # Stack in center
        description="C4: Center column battle"
    ))
    
    # Horizontal threat setup
    positions.append(TestPosition(
        move_sequence=[2, 0, 3, 1, 4],  # P1 building horizontal
        description="C4: Horizontal threat"
    ))
    
    # Alternating columns
    positions.append(TestPosition(
        move_sequence=[0, 1, 0, 1, 0, 1],
        description="C4: Alternating columns"
    ))
    
    return positions


def _othello_positions(logic) -> List[TestPosition]:
    """Othello-specific interesting positions."""
    positions = generate_test_positions(logic)
    
    # Othello often has pass situations - add positions likely to have limited moves
    # These would need to be validated against actual Othello rules
    
    return positions


def _gomoku_positions(logic) -> List[TestPosition]:
    """Gomoku-specific interesting positions."""
    positions = generate_test_positions(logic)
    
    # Add Gomoku-specific patterns if board is large enough
    board_size = logic.BOARD_SHAPE[0] if len(logic.BOARD_SHAPE) >= 1 else 15
    center = board_size // 2
    
    # Center play (convert 2D to 1D index)
    def idx(r, c):
        return r * board_size + c
    
    if board_size >= 9:
        positions.append(TestPosition(
            move_sequence=[idx(center, center)],
            description="Gomoku: Center opening"
        ))
        
        positions.append(TestPosition(
            move_sequence=[
                idx(center, center),
                idx(center, center+1),
                idx(center+1, center),
            ],
            description="Gomoku: Corner pattern"
        ))
    
    return positions


def _chess_positions(logic) -> List[TestPosition]:
    """
    Chess-specific test positions using FEN strings.
    Covers openings, middlegame, endgame, and special move types.
    """
    import chess
    from src.nn.kernels.chess_logic import chess_to_board

    positions = []

    # --- Generic positions (starting, after 1/2/3 moves, random, near-terminal) ---
    positions.extend(generate_test_positions(
        logic, num_random_positions=3, max_moves_for_random=40, seed=42
    ))

    # --- FEN-based positions ---
    fen_positions = [
        # Opening: Italian Game
        (
            "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
            "Chess: Italian Game (after 3. Bc4)",
        ),
        # Middlegame: complex with many pieces, castling rights lost
        (
            "r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 4 7",
            "Chess: Giuoco Piano middlegame",
        ),
        # Tactical: White has a fork opportunity
        (
            "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
            "Chess: Scholar's mate threat (Qh5)",
        ),
        # Endgame: Rook + pawn endgame (most common endgame type)
        (
            "8/5pk1/5p1p/8/1R6/6PP/5PK1/1r6 w - - 0 36",
            "Chess: Rook endgame",
        ),
        # Endgame: King + pawn vs King (key squares matter)
        (
            "8/8/4k3/8/4P3/4K3/8/8 w - - 0 1",
            "Chess: KP vs K endgame",
        ),
        # Position with en passant available
        (
            "rnbqkbnr/pppp1ppp/8/4pP2/8/8/PPPPP1PP/RNBQKBNR w KQkq e6 0 3",
            "Chess: En passant available",
        ),
        # Position where castling is the best move
        (
            "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
            "Chess: Castling available",
        ),
        # Promotion imminent
        (
            "8/3P1k2/8/8/8/8/5K2/8 w - - 0 1",
            "Chess: Pawn about to promote",
        ),
        # Many legal moves (queen in center, lots of pieces)
        (
            "r1b1k2r/ppppqppp/2n2n2/4p3/2BPP3/2N2Q2/PPP2PPP/R1B1K2R w KQkq - 0 7",
            "Chess: Complex middlegame (many legal moves)",
        ),
        # Few legal moves (cramped position)
        (
            "6k1/5ppp/8/8/8/8/5PPP/6K1 b - - 0 1",
            "Chess: Cramped position (few legal moves)",
        ),
    ]

    for fen, desc in fen_positions:
        try:
            cb = chess.Board(fen)
            if not cb.is_valid():
                continue
            board_arr = chess_to_board(cb)
            player = logic.PLAYER_1 if cb.turn == chess.WHITE else logic.PLAYER_2

            positions.append(TestPosition(
                move_sequence=[],
                description=desc,
                board=board_arr,
                player=player,
            ))
        except Exception:
            continue

    return positions


def validate_positions(logic, positions: List[TestPosition]) -> List[TestPosition]:
    """
    Validate that positions are actually playable and filter out invalid ones.
    """
    valid = []
    
    for pos in positions:
        try:
            board, player, done, winner = resolve_position(logic, pos)
            
            # Check if position matches expectations
            if done != pos.expected_game_over:
                # Game ended unexpectedly or didn't end when expected
                if done:
                    # Skip positions where game ended
                    continue
            
            # Verify there are legal moves (unless game is over)
            if not done:
                legal = get_legal_moves(logic, board, player)
                if not legal:
                    continue
            
            valid.append(pos)
            
        except Exception as e:
            # Skip positions that cause errors
            print(f"Warning: Skipping position '{pos.description}': {e}")
            continue
    
    return valid


def get_test_positions_for_game(logic, include_game_specific: bool = True) -> List[TestPosition]:
    """
    Main entry point: Get validated test positions for a game.
    
    Args:
        logic: Game logic module
        include_game_specific: Whether to include game-specific positions
        
    Returns:
        List of validated TestPosition objects
    """
    if include_game_specific:
        positions = generate_game_specific_positions(logic)
    else:
        positions = generate_test_positions(logic)
    
    return validate_positions(logic, positions)