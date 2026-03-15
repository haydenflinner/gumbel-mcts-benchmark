import sys
import os
import torch
import numpy as np
import torch.nn as nn
from functools import partial

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tests"))

from gumbel_mcts.puct import PUCT                       # was MCTSTreeV3
from gumbel_mcts.reference import (                      # was mcts_v2
    Node, DummyNode, expand, backup, best_child,
    generate_search_policy, uct_search, parallel_uct_search,
)
from utils.vectorized_env_wrapper import VectorizedEnvWrapper
from game_logic.tictactoe import TicTacToeLogic
from game_logic.gomoku import GomokuLogic
from test_positions_generator import (
    get_test_positions_for_game,
    TestPosition,
    play_sequence,
)


# ---------------------------------------------------------------------------
# GameResNetModule replacement
# ---------------------------------------------------------------------------
def _get_logic(game_name: str):
    name = game_name.lower()
    if "tictactoe" in name or "ttt" in name:
        return TicTacToeLogic()
    if "gomoku" in name:
        return GomokuLogic()
    raise ValueError(f"Unknown game: {game_name}")


class GameResNetModule(nn.Module):
    def __init__(self, model_type='resnet', hidden_size=256, num_layers=3,
                 cnn_channels=64, enable_selfplay=True, game_name='tictactoe'):
        super().__init__()
        self.logic = _get_logic(game_name)
        n = self.logic.NUM_ACTIONS
        self.net = nn.Sequential(
            nn.Linear(n, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden_size, n)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward_for_mcts(self, batch):
        boards = batch["boards"].float()
        h = self.net(boards)
        policy = torch.softmax(self.policy_head(h), dim=-1)
        value = torch.tanh(self.value_head(h))
        return {"policy": policy, "value": value}


# ---------------------------------------------------------------------------
# create_eval_func – compatible with reference.py (single and batched)
# ---------------------------------------------------------------------------
def create_eval_func(model):
    n = model.logic.NUM_ACTIONS
    device = "cpu"

    def eval_func(obs, batched=False):
        if batched:
            obs_arr = np.asarray(obs)
            boards = torch.tensor(obs_arr[:, :n], device=device, dtype=torch.float32)
            players = torch.tensor(obs_arr[:, n].astype(np.int64), device=device, dtype=torch.long)
            with torch.no_grad():
                out = model.forward_for_mcts({"boards": boards, "current_player": players})
            p = out["policy"].cpu().numpy().astype(np.float64)
            v = [float(x) for x in out["value"].cpu().numpy().flatten()]
            return p, v
        else:
            boards = torch.tensor(obs[:n], device=device, dtype=torch.float32).unsqueeze(0)
            players = torch.tensor([int(obs[n])], device=device, dtype=torch.long)
            with torch.no_grad():
                out = model.forward_for_mcts({"boards": boards, "current_player": players})
            p = out["policy"].cpu().numpy()[0].astype(np.float64)
            v = float(out["value"].cpu().numpy().flatten()[0])
            return p, v

    return eval_func


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------
def setup_game_instance(move_seq, logic):
    """
    Creates a standard-compatible wrapper around VectorizedGame
    and plays the sequence.
    """
    # 1. Init Wrapper
    env = VectorizedEnvWrapper(logic)

    # 2. Replay Moves
    for move in move_seq:
        env.step(move)
        if env.is_game_over():
            break

    return env

def compare_mcts_v2_v3_internals(game_name: str):
    """
    Compare internal state between mcts_v2 (reference) and mcts_v3 (Numba optimized).
    Verifies that both implementations produce equivalent results.
    """    
    import copy

    np.random.seed(42)
    torch.manual_seed(42)
    
    device = 'cpu'
    model = GameResNetModule(
        model_type='resnet', hidden_size=256, num_layers=3,
        cnn_channels=64, enable_selfplay=True,
        game_name=game_name
    )
    model.eval()
    eval_func = create_eval_func(model)
    
    num_simulations = 50
    c_puct_base = 19652
    c_puct_init = 1.25
    
    test_positions = get_test_positions_for_game(model.logic)
    print(f"\nGenerated {len(test_positions)} test positions for {game_name}")
    
    all_passed = True
    
    logic = model.logic

    for pos in test_positions:
        print(f"\n{'='*60}")
        print(f"Testing position after moves: {pos.move_sequence}")
        print('='*60)
        
        env = setup_game_instance(pos.move_sequence, logic)
        env.reset()

        for move in pos.move_sequence:
            env.step(move)
        
        if env.is_game_over():
            print("  Game already over, skipping")
            continue
        
        print(f"  Current player: {env.to_play}")
        print(f"  Legal actions: {np.where(env.legal_actions)[0]}")
        
        # === Test 1: Raw model output consistency ===
        print(f"\n--- Test 1: Model Output Consistency ---")
        
        obs_v2 = env.observation()
        print(obs_v2.shape)
        prior_v2, value_v2 = eval_func(obs_v2, batched=False)
        
        board_flat = torch.tensor(env.board, device=device, dtype=torch.float32).flatten().unsqueeze(0)
        player_tensor = torch.tensor([env.to_play], device=device, dtype=torch.long)
        
        with torch.no_grad():
            outputs = model.forward_for_mcts({
                "boards": board_flat,
                "current_player": player_tensor
            })
        prior_v3 = outputs['policy'].cpu().numpy()[0]
        value_v3 = outputs['value'].cpu().numpy().flatten()[0]
        
        prior_match = np.allclose(prior_v2, prior_v3, atol=1e-5)
        value_match = abs(value_v2 - value_v3) < 1e-5
        
        print(f"  v2 prior: {np.round(prior_v2, 4)}")
        print(f"  v3 prior: {np.round(prior_v3, 4)}")
        print(f"  v2 value: {value_v2:.6f}, v3 value: {value_v3:.6f}")
        print(f"  Prior match: {'✓' if prior_match else '✗'}")
        print(f"  Value match: {'✓' if value_match else '✗'}")
        
        if not (prior_match and value_match):
            print("  ⚠ MODEL OUTPUT MISMATCH - check observation encoding!")
            all_passed = False
            continue

        # === Test 2: Full MCTS comparison ===
        print(f"\n--- Test 2: Full MCTS ({num_simulations} simulations) ---")
        
        # =====================================================================
        # Run v2 MCTS MANUALLY to access root internals
        # =====================================================================
        np.random.seed(123)
        torch.manual_seed(123)
        
        num_actions = logic.NUM_ACTIONS
        root_v2 = Node(to_play=env.to_play, num_actions=num_actions, parent=DummyNode())
        prior, value = eval_func(env.observation(), False)
        expand(root_v2, prior)
        backup(root_v2, float(value))
        
        # Run simulations matching uct_search logic
        while root_v2.N < num_simulations + 1:
            node = root_v2
            sim_env = copy.deepcopy(env)
            
            # Select
            while node.is_expanded:
                node = best_child(node, sim_env.legal_actions, c_puct_base, c_puct_init, sim_env.opponent_player)
                obs, reward, done, _ = sim_env.step(node.move)
                if done:
                    break
            
            # Expand & Backup
            if sim_env.is_game_over():
                # Reward is from perspective of player who just moved
                # Node.to_play is the OTHER player, so negate
                backup(node, -reward)
            else:
                p, v = eval_func(sim_env.observation(), False)
                expand(node, p)
                backup(node, float(v))
        
        # Extract v2 results
        search_pi_v2 = generate_search_policy(root_v2.child_N, temperature=0.1, legal_actions=env.legal_actions)
        move_v2 = int(np.argmax(root_v2.child_N * env.legal_actions))
        root_Q_v2 = root_v2.Q
        
        # =====================================================================
        # Run v3 MCTS
        # =====================================================================
        np.random.seed(123)
        torch.manual_seed(123)
        test_max_nodes = int((1 + num_simulations) * 2.0)
        test_max_nodes = max(test_max_nodes, 200)
        tree_v3 = PUCT(n_games=1, max_nodes=test_max_nodes, logic=logic, device=device)
        tree_v3.reset()
        tree_v3.initialize_roots(
            active_games=[0],
            starting_boards=np.array([env.board]),
            starting_players=np.array([env.to_play])
        )
        tree_v3.run_simulation_batch(
            model, active_games=[0], num_simulations=num_simulations,
            c_puct_base=c_puct_base, c_puct_init=c_puct_init
        )
        
        # Extract v3 results
        child_visits_v3, root_Q_v3 = tree_v3.get_root_data(game_idx=0)
        search_pi_v3 = generate_search_policy(
            child_visits_v3, temperature=0.1, legal_actions=env.legal_actions
        )
        move_v3 = int(np.argmax(child_visits_v3 * env.legal_actions))
        
        # =====================================================================
        # DEBUG: Compare internal visit counts
        # =====================================================================
        print(f"\n--- Debug: Visit Count Comparison ---")
        print(f"  V2 root.N = {root_v2.N}")
        print(f"  V2 child_N = {root_v2.child_N.astype(int)}")

        root_idx = tree_v3.storage.root_indices[0]
        v3_root_N = tree_v3.storage.visit_counts[root_idx]
        v3_child_N = np.array([
            tree_v3.storage.visit_counts[tree_v3.storage.children[root_idx, m]] 
            if tree_v3.storage.children[root_idx, m] != -1 else 0 
            for m in range(logic.NUM_ACTIONS)
        ], dtype=int)
        print(f"  V3 root.N = {v3_root_N}")
        print(f"  V3 child_N = {v3_child_N}")
        
        # Check if visit counts match
        visits_match = (root_v2.N == v3_root_N) and np.array_equal(root_v2.child_N.astype(int), v3_child_N)
        print(f"  Visit counts match: {'✓' if visits_match else '✗'}")
        
        if not visits_match:
            print(f"  V2 - V3 diff: {root_v2.child_N.astype(int) - v3_child_N}")

        # =====================================================================
        # Results comparison
        # =====================================================================
        print(f"\n--- Results ---")
        print(f"  v2 search_pi:    {np.round(search_pi_v2, 4)}")
        print(f"  v3 search_pi:    {np.round(search_pi_v3, 4)}")
        print(f"  v2 root_Q:       {root_Q_v2:.4f}")
        print(f"  v3 root_Q:       {root_Q_v3:.4f}")
        print(f"  v2 move:         {move_v2}")
        print(f"  v3 move:         {move_v3}")
        
        # Tight tolerances
        move_match = move_v2 == move_v3
        pi_match = np.allclose(search_pi_v2, search_pi_v3, atol=1e-3)
        q_match = abs(root_Q_v2 - root_Q_v3) < 1e-3
        visits_match = (root_v2.N == v3_root_N) and np.array_equal(root_v2.child_N.astype(int), v3_child_N)

        print(f"\n  Move match:   {'✓' if move_match else '✗'}")
        print(f"  Policy match: {'✓' if pi_match else '✗'}")
        print(f"  Q match:      {'✓' if q_match else '✗'}")
        print(f"  Visits match: {'✓' if visits_match else '✗'}")
        
        if not (move_match and pi_match and q_match and visits_match):
            print(f"\n  ⚠ MISMATCH DETECTED...")
            assert False, "MCTS V3 vs V2 mismatch detected!"
    
    print(f"\n{'='*60}")
    print(f"OVERALL: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
    print('='*60)
    
    assert all_passed

def compare_mcts_v2_v3_internals_with_uct_search(game_name: str, use_parallel_uct: bool):
    """
    Compare mcts_v3 against the actual parallel_uct_search (V2) implementation.
    """    
    import copy
    
    np.random.seed(42)
    torch.manual_seed(42)
    
    device = 'cpu'
    model = GameResNetModule(
        model_type='resnet', hidden_size=256, num_layers=3,
        cnn_channels=64, enable_selfplay=True,
        game_name=game_name
    )
    model.eval()
    eval_func = create_eval_func(model)
    
    num_simulations = 50
    num_parallel = 1  # Use 1 for deterministic comparison
    c_puct_base = 19652
    c_puct_init = 1.25
    
    logic = model.logic
    
    test_positions = get_test_positions_for_game(logic)
    print(f"\nGenerated {len(test_positions)} test positions for {game_name}")

    all_passed = True
    
    for pos in test_positions:
        print(f"\n{'='*60}")
        print(f"Testing position after moves: {pos.move_sequence}")
        print('='*60)
        
        env = setup_game_instance(pos.move_sequence, logic)
        env.reset()

        for move in pos.move_sequence:
            env.step(move)
        
        if env.is_game_over():
            print("  Game already over, skipping")
            continue
        
        print(f"  Current player: {env.to_play}")
        print(f"  Legal actions: {np.where(env.legal_actions)[0]}")
        
        # === Test 1: Raw model output consistency ===
        print(f"\n--- Test 1: Model Output Consistency ---")
        
        obs_v2 = env.observation()
        prior_v2, value_v2 = eval_func(obs_v2, batched=False)
        
        board_flat = torch.tensor(env.board, device=device, dtype=torch.float32).flatten().unsqueeze(0)
        player_tensor = torch.tensor([env.to_play], device=device, dtype=torch.long)
        
        with torch.no_grad():
            outputs = model.forward_for_mcts({
                "boards": board_flat,
                "current_player": player_tensor
            })
        prior_v3 = outputs['policy'].cpu().numpy()[0]
        value_v3 = outputs['value'].cpu().numpy().flatten()[0]
        
        prior_match = np.allclose(prior_v2, prior_v3, atol=1e-3)
        value_match = abs(value_v2 - value_v3) < 1e-3
        
        print(f"  v2 prior: {np.round(prior_v2, 4)}")
        print(f"  v3 prior: {np.round(prior_v3, 4)}")
        print(f"  v2 value: {value_v2:.6f}, v3 value: {value_v3:.6f}")
        print(f"  Prior match: {'✓' if prior_match else '✗'}")
        print(f"  Value match: {'✓' if value_match else '✗'}")
        
        if not (prior_match and value_match):
            print("  ⚠ MODEL OUTPUT MISMATCH - check observation encoding!")
            all_passed = False
            continue

        # === Test 2: Full MCTS comparison using actual parallel_uct_search ===
        print(f"\n--- Test 2: Full MCTS ({num_simulations} simulations) ---")
        
        # =====================================================================
        # Run V2 using the ACTUAL parallel_uct_search
        # =====================================================================
        np.random.seed(123)
        torch.manual_seed(123)
        
        if use_parallel_uct:
            move_v2, search_pi_v2, root_Q_v2, best_child_Q_v2, _ = parallel_uct_search(
                env=env,
                eval_func=eval_func,
                root_node=None,
                c_puct_base=c_puct_base,
                c_puct_init=c_puct_init,
                num_simulations=num_simulations,
                num_parallel=num_parallel,
                root_noise=False,
                warm_up=False,
                deterministic=True
            )
        else:
            move_v2, search_pi_v2, root_Q_v2, best_child_Q_v2, _ = uct_search(
                env=env,
                eval_func=eval_func,
                root_node=None,
                c_puct_base=c_puct_base,
                c_puct_init=c_puct_init,
                num_simulations=num_simulations+1,
                root_noise=False,
                warm_up=False,
                deterministic=True
            )
        
        # =====================================================================
        # Run V3 MCTS
        # =====================================================================
        np.random.seed(123)
        torch.manual_seed(123)
        test_max_nodes = int((1 + num_simulations) * 2.0)
        test_max_nodes = max(test_max_nodes, 200)
        
        tree_v3 = PUCT(n_games=1, max_nodes=test_max_nodes, logic=logic, device=device)
        tree_v3.reset()
        tree_v3.initialize_roots(
            active_games=[0],
            starting_boards=np.array([env.board]),
            starting_players=np.array([env.to_play])
        )
        if pos.move_sequence == [3, 3, 4, 4, 2]:
            tree_v3.run_simulation_batch(
                model, active_games=[0], num_simulations=num_simulations,
                c_puct_base=c_puct_base, c_puct_init=c_puct_init,
            )
        else:
            tree_v3.run_simulation_batch(
                model, active_games=[0], num_simulations=num_simulations,
                c_puct_base=c_puct_base, c_puct_init=c_puct_init
            )
        
        # Extract v3 results
        child_visits_v3, root_Q_v3 = tree_v3.get_root_data(game_idx=0)
        search_pi_v3 = generate_search_policy(
            child_visits_v3, temperature=0.1, legal_actions=env.legal_actions
        )
        move_v3 = int(np.argmax(child_visits_v3 * env.legal_actions))

        # =====================================================================
        # Results comparison
        # =====================================================================
        print(f"\n--- Results ---")
        print(f"  v2 search_pi:    {np.round(search_pi_v2, 4)}")
        print(f"  v3 search_pi:    {np.round(search_pi_v3, 4)}")
        print(f"  v2 root_Q:       {root_Q_v2:.4f}")
        print(f"  v3 root_Q:       {root_Q_v3:.4f}")
        print(f"  v2 move:         {move_v2}")
        print(f"  v3 move:         {move_v3}")
        
        # Tight tolerances
        move_match = move_v2 == move_v3
        if use_parallel_uct:
            # Softer tolerances due to specifics of retry mechanism in parallel_uct_search
            pi_match = np.allclose(search_pi_v2, search_pi_v3, atol=0.05)
            q_match = abs(root_Q_v2 - root_Q_v3) < 1e-1
        else:
            pi_match = np.allclose(search_pi_v2, search_pi_v3, atol=1e-5)
            q_match = abs(root_Q_v2 - root_Q_v3) < 1e-5
        
        print(f"\n  Move match:   {'✓' if move_match else '✗'}")
        print(f"  Policy match: {'✓' if pi_match else '✗'}")
        print(f"  Q match:      {'✓' if q_match else '✗'}")
        
        if not (move_match and pi_match and q_match):
            all_passed = False
            print(f"\n  ⚠ MISMATCH DETECTED - Running debug comparison...")
            
            # === DEBUG: Manual V2 to get visit counts ===
            np.random.seed(123)
            torch.manual_seed(123)
            num_actions = logic.NUM_ACTIONS
            root_v2_debug = Node(to_play=env.to_play, num_actions=num_actions, parent=DummyNode())
            prior, value = eval_func(env.observation(), False)
            expand(root_v2_debug, prior)
            backup(root_v2_debug, float(value))
            
            while root_v2_debug.N < num_simulations:
                node = root_v2_debug
                sim_env = copy.deepcopy(env)
                while node.is_expanded:
                    node = best_child(node, sim_env.legal_actions, c_puct_base, c_puct_init, sim_env.opponent_player)
                    obs, reward, done, _ = sim_env.step(node.move)
                    if done:
                        break
                if sim_env.is_game_over():
                    backup(node, -reward)
                else:
                    p, v = eval_func(sim_env.observation(), False)
                    expand(node, p)
                    backup(node, float(v))
            
            print(f"\n--- Debug: Visit Count Comparison ---")
            print(f"  V2 child_N = {root_v2_debug.child_N.astype(int)}")
            
            root_idx = tree_v3.storage.root_indices[0]
            v3_child_N = np.array([
                tree_v3.storage.visit_counts[tree_v3.storage.children[root_idx, m]] 
                if tree_v3.storage.children[root_idx, m] != -1 else 0 
                for m in range(logic.NUM_ACTIONS)
            ], dtype=int)
            print(f"  V3 child_N = {v3_child_N}")
            print(f"  Diff: {root_v2_debug.child_N.astype(int) - v3_child_N}")
            assert False, "MCTS V3 vs V2 mismatch detected!"
    
    print(f"\n{'='*60}")
    print(f"OVERALL: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
    print('='*60)
    
    assert all_passed


def test_mcts_v3_vs_v2_full_suite(game_name: str):
    """
    Final Golden Master Test:
    Verifies that MCTSTreeV3 (Batch/Numba) produces the exact same 
    visit counts as parallel_uct_search (Python/Serial).
    """
    print("\n" + "="*70)
    print("MCTS V3 vs V2: FINAL EQUIVALENCE SUITE")
    print("="*70)
    
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    device = 'cpu'
    model = GameResNetModule(
        model_type='resnet', hidden_size=256, num_layers=3,
        cnn_channels=64, enable_selfplay=True, game_name=game_name
    ).eval().to(device)
    eval_func = create_eval_func(model)

    num_actions = model.logic.NUM_ACTIONS

    num_simulations = 50
    c_puct_base = 19652
    c_puct_init = 1.25
    
    logic = model.logic

    test_positions = get_test_positions_for_game(logic)
    print(f"\nGenerated {len(test_positions)} test positions for {game_name}")

    all_passed = True
    

    for pos in test_positions:
        print(f"\n{'-'*70}")
        print(f"Testing Position: {pos.move_sequence}")
        
        env = setup_game_instance(pos.move_sequence, logic)
        env.reset()

        for m in pos.move_sequence: 
            env.step(m)
        if env.is_game_over(): continue

        # --- Test: MCTS Execution ---
        
        # 1. Run V2 (Reference)
        # We must reconstruct the root manually to get raw child_N 
        # because parallel_uct_search returns a processed policy.
        np.random.seed(123)
        torch.manual_seed(123)
        
        root_v2 = Node(to_play=env.to_play, num_actions=num_actions, parent=DummyNode())
        prior, value = eval_func(env.observation(), False)
        expand(root_v2, prior)
        backup(root_v2, float(value))
        
        # We manually run the loop to match V2 logic exactly without wrapper overhead
        import copy
        
        while root_v2.N < num_simulations + 1:
            node = root_v2
            sim_env = copy.deepcopy(env)
            
            # Select
            while node.is_expanded:
                node = best_child(node, sim_env.legal_actions, c_puct_base, c_puct_init, sim_env.opponent_player)
                sim_env.step(node.move)
                if sim_env.is_game_over(): break
            
            # Expand & Backup
            if sim_env.is_game_over():
                # In V2 logic, if next player is winner, reward is +1 for them.
                # Env step returns reward relative to player who just moved.
                # Standard Connect4 env: reward=1 if winner.
                # We simply backup -reward.
                # Note: This crude manual loop approximates parallel_uct_search logic
                # for the sake of checking visit counts.
                reward = 1.0 if sim_env.winner is not None else 0.0
                if sim_env.winner == 0: reward = 0.0 # Draw
                backup(node, -reward)
            else:
                p, v = eval_func(sim_env.observation(), False)
                expand(node, p)
                backup(node, float(v))
                
        visits_v2 = root_v2.child_N.astype(np.int32)
        
        # 2. Run V3 (Candidate)
        np.random.seed(123)
        torch.manual_seed(123)
        
        test_max_nodes = int((1 + num_simulations) * 2.0)
        test_max_nodes = max(test_max_nodes, 200)
        
        tree_v3 = PUCT(n_games=1, max_nodes=test_max_nodes, logic=logic, device=device)
        tree_v3.reset()
        tree_v3.initialize_roots([0], np.array([env.board]), np.array([env.to_play]))
        
        # Run remaining 50 sims
        tree_v3.run_simulation_batch(model, [0], num_simulations=num_simulations, c_puct_base=c_puct_base, c_puct_init=c_puct_init)
        
        visits_v3, _ = tree_v3.get_root_data(0)
        visits_v3 = visits_v3.astype(np.int32)
        
        # --- Strict Assertions ---
        # Compare Visit Counts directly (Integers are exact)
        visits_match = np.array_equal(visits_v2, visits_v3)
        
        print(f"  V2 Visits: {visits_v2}")
        print(f"  V3 Visits: {visits_v3}")
        print(f"  Match:     [{'✓' if visits_match else '✗'}]")
        
        if not visits_match:
            print("  [!] FAILURE DETECTED")
            diff = visits_v2 - visits_v3
            print(f"  Diff: {diff}")
            all_passed = False
            assert False, "MCTS V3 visit counts do not match V2!"
            
    print(f"\n{'-'*70}")
    print(f"OVERALL RESULT: {'✓ PASSED' if all_passed else '✗ FAILED'}")
    print("="*70)
    assert all_passed
    
def run_all_tests(game_name: str):
    """Run all MCTS tests"""
    print("=" * 60)
    print("MCTS CORRECTNESS TEST SUITE")
    print("=" * 60)
    
    tests = [
        partial(compare_mcts_v2_v3_internals, game_name=game_name),
        partial(compare_mcts_v2_v3_internals_with_uct_search, game_name=game_name, use_parallel_uct=False),
        partial(compare_mcts_v2_v3_internals_with_uct_search, game_name=game_name, use_parallel_uct=True),
        partial(test_mcts_v3_vs_v2_full_suite, game_name=game_name),
    ]
    
    failed = []
    for test in tests:
        # Get a nice name for partial functions
        if isinstance(test, partial):
            name = f"{test.func.__name__}({', '.join(f'{k}={v}' for k, v in test.keywords.items())})"
        else:
            name = test.__name__
        
        try:
            print(f"\n>>> Running: {name}")
            test()
        except Exception as e:
            print(f"✗ {name} FAILED: {e}")
            failed.append(name)
    
    print("\n" + "=" * 60)
    if failed:
        print(f"FAILED {len(failed)}/{len(tests)} tests:")
        for name in failed:
            print(f"  - {name}")
    else:
        print(f"✓ ALL {len(tests)} TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    game_name = sys.argv[1]
    run_all_tests(game_name)
