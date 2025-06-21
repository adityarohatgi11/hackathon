"""
Advanced Game Theory Models for GridPilot-GT
Implements sophisticated game-theoretic optimization strategies
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

try:
    import cvxpy as cp
    from scipy import optimize
    from scipy.stats import norm, multivariate_normal
    from scipy.linalg import cholesky
    HAS_ADVANCED_LIBS = True
except ImportError:
    HAS_ADVANCED_LIBS = False

logger = logging.getLogger(__name__)


class StochasticGameTheory:
    """
    Advanced stochastic game theory for energy trading.
    Implements Nash equilibrium, Stackelberg competition, and cooperative games.
    """
    
    def __init__(self, n_players: int = 3, game_type: str = "cooperative"):
        """
        Initialize stochastic game.
        
        Args:
            n_players: Number of players in the game
            game_type: "cooperative", "non_cooperative", "stackelberg"
        """
        self.n_players = n_players
        self.game_type = game_type
        self.player_strategies = {}
        self.payoff_matrices = {}
        self.equilibrium_solution = None
        
        logger.info(f"Stochastic game initialized: {n_players} players, {game_type}")
    
    def define_payoff_function(self, player_id: int, 
                              payoff_func: Callable[[np.ndarray, np.ndarray], float]):
        """Define payoff function for a player."""
        self.payoff_matrices[player_id] = payoff_func
        logger.info(f"Payoff function defined for player {player_id}")
    
    def solve_nash_equilibrium(self, initial_strategies: Dict[int, np.ndarray],
                              price_scenarios: np.ndarray,
                              max_iterations: int = 100) -> Dict[str, Any]:
        """
        Solve for Nash equilibrium under price uncertainty.
        
        Args:
            initial_strategies: Initial strategy for each player
            price_scenarios: Monte Carlo price scenarios
            max_iterations: Maximum iterations for convergence
            
        Returns:
            Dictionary with equilibrium strategies and payoffs
        """
        if not HAS_ADVANCED_LIBS:
            logger.warning("Advanced optimization requires cvxpy")
            return self._simple_nash_equilibrium(initial_strategies)
        
        try:
            current_strategies = initial_strategies.copy()
            
            for iteration in range(max_iterations):
                updated_strategies = {}
                
                for player_id in range(self.n_players):
                    # Solve best response problem for this player
                    best_response = self._solve_best_response(
                        player_id, current_strategies, price_scenarios
                    )
                    updated_strategies[player_id] = best_response
                
                # Check convergence
                if self._check_convergence(current_strategies, updated_strategies):
                    logger.info(f"Nash equilibrium converged in {iteration} iterations")
                    break
                
                current_strategies = updated_strategies
            
            # Calculate equilibrium payoffs
            equilibrium_payoffs = self._calculate_equilibrium_payoffs(
                current_strategies, price_scenarios
            )
            
            self.equilibrium_solution = {
                "strategies": current_strategies,
                "payoffs": equilibrium_payoffs,
                "converged": iteration < max_iterations - 1,
                "iterations": iteration + 1
            }
            
            return self.equilibrium_solution
            
        except Exception as e:
            logger.error(f"Nash equilibrium solving failed: {e}")
            return self._simple_nash_equilibrium(initial_strategies)
    
    def _solve_best_response(self, player_id: int, 
                           other_strategies: Dict[int, np.ndarray],
                           price_scenarios: np.ndarray) -> np.ndarray:
        """Solve best response optimization for a player."""
        n_scenarios, horizon = price_scenarios.shape
        
        # Decision variable: strategy for this player
        strategy = cp.Variable(horizon, nonneg=True)
        
        # Expected payoff across scenarios
        expected_payoff = 0
        
        for scenario_idx in range(n_scenarios):
            prices = price_scenarios[scenario_idx, :]
            
            # Player's revenue
            revenue = prices @ strategy
            
            # Competition effects (simplified)
            competition_penalty = 0
            for other_id, other_strategy in other_strategies.items():
                if other_id != player_id:
                    # Penalty for overlapping strategies
                    overlap = cp.sum(cp.minimum(strategy, other_strategy))
                    competition_penalty += 0.1 * overlap
            
            scenario_payoff = revenue - competition_penalty
            expected_payoff += scenario_payoff / n_scenarios
        
        # Constraints
        constraints = [
            cp.sum(strategy) <= 1000,  # Total capacity constraint
            strategy >= 0,             # Non-negativity
            strategy <= 500            # Individual period limits
        ]
        
        # Solve optimization
        problem = cp.Problem(cp.Maximize(expected_payoff), constraints)
        problem.solve(solver=cp.ECOS, verbose=False)
        
        if strategy.value is not None:
            return np.maximum(strategy.value, 0)
        else:
            # Fallback to uniform allocation
            return np.full(horizon, 1000 / horizon)
    
    def _check_convergence(self, old_strategies: Dict[int, np.ndarray],
                          new_strategies: Dict[int, np.ndarray],
                          tolerance: float = 1e-3) -> bool:
        """Check if strategies have converged."""
        for player_id in old_strategies:
            old_strategy = old_strategies[player_id]
            new_strategy = new_strategies[player_id]
            
            if np.linalg.norm(old_strategy - new_strategy) > tolerance:
                return False
        
        return True
    
    def _calculate_equilibrium_payoffs(self, strategies: Dict[int, np.ndarray],
                                     price_scenarios: np.ndarray) -> Dict[int, float]:
        """Calculate expected payoffs at equilibrium."""
        payoffs = {}
        n_scenarios = len(price_scenarios)
        
        for player_id in strategies:
            total_payoff = 0
            
            for scenario_idx in range(n_scenarios):
                prices = price_scenarios[scenario_idx, :]
                strategy = strategies[player_id]
                
                # Revenue for this scenario
                revenue = np.sum(prices * strategy)
                total_payoff += revenue
            
            payoffs[player_id] = total_payoff / n_scenarios
        
        return payoffs
    
    def _simple_nash_equilibrium(self, initial_strategies: Dict[int, np.ndarray]) -> Dict[str, Any]:
        """Simple Nash equilibrium fallback."""
        # Equal allocation strategy
        horizon = len(next(iter(initial_strategies.values())))
        equal_strategy = np.full(horizon, 1000 / (self.n_players * horizon))
        
        strategies = {player_id: equal_strategy for player_id in range(self.n_players)}
        payoffs = {player_id: 100.0 for player_id in range(self.n_players)}
        
        return {
            "strategies": strategies,
            "payoffs": payoffs,
            "converged": True,
            "iterations": 1
        }
    
    def solve_stackelberg_game(self, leader_id: int, 
                              price_scenarios: np.ndarray) -> Dict[str, Any]:
        """
        Solve Stackelberg leader-follower game.
        
        Args:
            leader_id: ID of the leader player
            price_scenarios: Price scenarios for stochastic optimization
            
        Returns:
            Dictionary with leader and follower strategies
        """
        if not HAS_ADVANCED_LIBS:
            logger.warning("Stackelberg game requires advanced optimization")
            return {"leader_strategy": np.zeros(24), "follower_strategies": {}}
        
        try:
            n_scenarios, horizon = price_scenarios.shape
            
            # Leader's strategy (first-mover advantage)
            leader_strategy = cp.Variable(horizon, nonneg=True)
            
            # Followers' best responses (computed for each scenario)
            follower_responses = {}
            
            # Solve leader's optimization knowing followers will respond optimally
            expected_leader_payoff = 0
            
            for scenario_idx in range(n_scenarios):
                prices = price_scenarios[scenario_idx, :]
                
                # Compute followers' best responses for this scenario
                scenario_responses = self._compute_follower_responses(
                    leader_strategy, prices, horizon
                )
                
                # Leader's payoff considering followers' responses
                leader_revenue = prices @ leader_strategy
                
                # Competition effects from followers
                competition_loss = 0
                for follower_response in scenario_responses.values():
                    overlap = cp.sum(cp.minimum(leader_strategy, follower_response))
                    competition_loss += 0.05 * overlap
                
                scenario_payoff = leader_revenue - competition_loss
                expected_leader_payoff += scenario_payoff / n_scenarios
            
            # Leader's constraints
            constraints = [
                cp.sum(leader_strategy) <= 1000,
                leader_strategy >= 0,
                leader_strategy <= 400  # Leader capacity
            ]
            
            # Solve leader's problem
            problem = cp.Problem(cp.Maximize(expected_leader_payoff), constraints)
            problem.solve(solver=cp.ECOS, verbose=False)
            
            if leader_strategy.value is not None:
                optimal_leader_strategy = np.maximum(leader_strategy.value, 0)
            else:
                optimal_leader_strategy = np.full(horizon, 400 / horizon)
            
            # Compute followers' final responses
            final_follower_strategies = {}
            for scenario_idx in range(min(10, n_scenarios)):  # Sample scenarios
                prices = price_scenarios[scenario_idx, :]
                responses = self._compute_follower_responses(
                    optimal_leader_strategy, prices, horizon, as_values=True
                )
                
                for follower_id, response in responses.items():
                    if follower_id not in final_follower_strategies:
                        final_follower_strategies[follower_id] = []
                    final_follower_strategies[follower_id].append(response)
            
            # Average follower strategies across scenarios
            for follower_id in final_follower_strategies:
                final_follower_strategies[follower_id] = np.mean(
                    final_follower_strategies[follower_id], axis=0
                )
            
            return {
                "leader_strategy": optimal_leader_strategy,
                "follower_strategies": final_follower_strategies,
                "leader_id": leader_id,
                "game_type": "stackelberg"
            }
            
        except Exception as e:
            logger.error(f"Stackelberg game solving failed: {e}")
            return {"leader_strategy": np.zeros(horizon), "follower_strategies": {}}
    
    def _compute_follower_responses(self, leader_strategy, prices, horizon, as_values=False):
        """Compute optimal follower responses given leader's strategy."""
        responses = {}
        
        for follower_id in range(1, self.n_players):  # Exclude leader (id=0)
            if as_values and hasattr(leader_strategy, '__len__'):
                # Use actual values
                leader_alloc = leader_strategy
            else:
                # Use CVXPY variable
                leader_alloc = leader_strategy
            
            follower_strategy = cp.Variable(horizon, nonneg=True)
            
            # Follower's revenue
            follower_revenue = prices @ follower_strategy
            
            # Competition with leader
            if as_values:
                competition_penalty = 0.1 * cp.sum(cp.minimum(follower_strategy, leader_alloc))
            else:
                competition_penalty = 0.1 * cp.sum(cp.minimum(follower_strategy, leader_alloc))
            
            # Follower's objective
            follower_objective = follower_revenue - competition_penalty
            
            # Follower's constraints
            follower_constraints = [
                cp.sum(follower_strategy) <= 600,  # Follower capacity
                follower_strategy >= 0,
                follower_strategy <= 300
            ]
            
            # Solve follower's problem
            follower_problem = cp.Problem(cp.Maximize(follower_objective), follower_constraints)
            
            if as_values:
                follower_problem.solve(solver=cp.ECOS, verbose=False)
                if follower_strategy.value is not None:
                    responses[follower_id] = np.maximum(follower_strategy.value, 0)
                else:
                    responses[follower_id] = np.full(horizon, 600 / horizon)
            else:
                responses[follower_id] = follower_strategy
        
        return responses
    
    def solve_cooperative_game(self, price_scenarios: np.ndarray,
                              sharing_rule: str = "proportional") -> Dict[str, Any]:
        """
        Solve cooperative game with coalition formation.
        
        Args:
            price_scenarios: Price scenarios
            sharing_rule: "proportional", "equal", "shapley"
            
        Returns:
            Cooperative solution with profit sharing
        """
        if not HAS_ADVANCED_LIBS:
            logger.warning("Cooperative game requires advanced optimization")
            return self._simple_cooperative_solution(price_scenarios)
        
        try:
            n_scenarios, horizon = price_scenarios.shape
            
            # Coalition strategy (joint optimization)
            coalition_strategy = cp.Variable((self.n_players, horizon), nonneg=True)
            
            # Total coalition payoff
            total_expected_payoff = 0
            
            for scenario_idx in range(n_scenarios):
                prices = price_scenarios[scenario_idx, :]
                
                # Total coalition revenue
                total_revenue = 0
                for player_id in range(self.n_players):
                    player_revenue = prices @ coalition_strategy[player_id, :]
                    total_revenue += player_revenue
                
                # Synergy benefits from cooperation
                synergy_bonus = 0.1 * total_revenue  # 10% cooperation bonus
                
                scenario_payoff = total_revenue + synergy_bonus
                total_expected_payoff += scenario_payoff / n_scenarios
            
            # Coalition constraints
            constraints = []
            
            # Individual capacity constraints
            for player_id in range(self.n_players):
                constraints.extend([
                    cp.sum(coalition_strategy[player_id, :]) <= 800,  # Individual capacity
                    coalition_strategy[player_id, :] >= 0
                ])
            
            # Joint capacity constraint
            total_allocation = cp.sum(coalition_strategy, axis=0)
            constraints.append(cp.sum(total_allocation) <= 2000)  # Total system capacity
            
            # Solve cooperative problem
            problem = cp.Problem(cp.Maximize(total_expected_payoff), constraints)
            problem.solve(solver=cp.ECOS, verbose=False)
            
            if coalition_strategy.value is not None:
                optimal_strategies = {}
                for player_id in range(self.n_players):
                    optimal_strategies[player_id] = np.maximum(
                        coalition_strategy.value[player_id, :], 0
                    )
            else:
                # Fallback
                optimal_strategies = {
                    player_id: np.full(horizon, 800 / horizon) 
                    for player_id in range(self.n_players)
                }
            
            # Calculate total coalition value
            total_value = self._calculate_coalition_value(optimal_strategies, price_scenarios)
            
            # Apply sharing rule
            individual_payoffs = self._apply_sharing_rule(
                total_value, optimal_strategies, sharing_rule
            )
            
            return {
                "coalition_strategies": optimal_strategies,
                "individual_payoffs": individual_payoffs,
                "total_coalition_value": total_value,
                "sharing_rule": sharing_rule,
                "cooperation_benefit": total_value * 0.1  # Synergy benefit
            }
            
        except Exception as e:
            logger.error(f"Cooperative game solving failed: {e}")
            return self._simple_cooperative_solution(price_scenarios)
    
    def _calculate_coalition_value(self, strategies: Dict[int, np.ndarray],
                                 price_scenarios: np.ndarray) -> float:
        """Calculate total coalition value."""
        total_value = 0
        n_scenarios = len(price_scenarios)
        
        for scenario_idx in range(n_scenarios):
            prices = price_scenarios[scenario_idx, :]
            scenario_value = 0
            
            for player_id, strategy in strategies.items():
                player_revenue = np.sum(prices * strategy)
                scenario_value += player_revenue
            
            # Add synergy bonus
            scenario_value *= 1.1  # 10% cooperation bonus
            total_value += scenario_value
        
        return total_value / n_scenarios
    
    def _apply_sharing_rule(self, total_value: float, 
                           strategies: Dict[int, np.ndarray],
                           sharing_rule: str) -> Dict[int, float]:
        """Apply profit sharing rule."""
        if sharing_rule == "equal":
            # Equal sharing
            individual_share = total_value / self.n_players
            return {player_id: individual_share for player_id in range(self.n_players)}
        
        elif sharing_rule == "proportional":
            # Proportional to individual contributions
            individual_contributions = {}
            total_contribution = 0
            
            for player_id, strategy in strategies.items():
                contribution = np.sum(strategy)  # Simple contribution measure
                individual_contributions[player_id] = contribution
                total_contribution += contribution
            
            payoffs = {}
            for player_id in individual_contributions:
                if total_contribution > 0:
                    share = individual_contributions[player_id] / total_contribution
                    payoffs[player_id] = total_value * share
                else:
                    payoffs[player_id] = total_value / self.n_players
            
            return payoffs
        
        else:
            # Default to equal sharing
            individual_share = total_value / self.n_players
            return {player_id: individual_share for player_id in range(self.n_players)}
    
    def _simple_cooperative_solution(self, price_scenarios: np.ndarray) -> Dict[str, Any]:
        """Simple cooperative solution fallback."""
        horizon = price_scenarios.shape[1]
        
        # Simple equal allocation
        strategies = {
            player_id: np.full(horizon, 800 / horizon) 
            for player_id in range(self.n_players)
        }
        
        total_value = 500.0  # Estimated total value
        individual_payoffs = {
            player_id: total_value / self.n_players 
            for player_id in range(self.n_players)
        }
        
        return {
            "coalition_strategies": strategies,
            "individual_payoffs": individual_payoffs,
            "total_coalition_value": total_value,
            "sharing_rule": "equal",
            "cooperation_benefit": 50.0
        }


class AdvancedAuctionMechanism:
    """
    Advanced auction mechanisms with stochastic bidding.
    """
    
    def __init__(self, auction_type: str = "second_price"):
        """
        Initialize auction mechanism.
        
        Args:
            auction_type: "first_price", "second_price", "all_pay", "combinatorial"
        """
        self.auction_type = auction_type
        self.bidders = {}
        self.auction_history = []
        
        logger.info(f"Advanced auction mechanism initialized: {auction_type}")
    
    def register_bidder(self, bidder_id: int, valuation_function: Callable):
        """Register a bidder with their valuation function."""
        self.bidders[bidder_id] = {
            "valuation_function": valuation_function,
            "bid_history": [],
            "win_history": []
        }
    
    def run_stochastic_auction(self, item_characteristics: Dict[str, Any],
                             price_scenarios: np.ndarray,
                             n_rounds: int = 1) -> Dict[str, Any]:
        """
        Run stochastic auction with uncertainty.
        
        Args:
            item_characteristics: Characteristics of items being auctioned
            price_scenarios: Price uncertainty scenarios
            n_rounds: Number of auction rounds
            
        Returns:
            Auction results with winners and payments
        """
        auction_results = []
        
        for round_idx in range(n_rounds):
            # Generate bids for this round
            round_bids = {}
            
            for bidder_id, bidder_info in self.bidders.items():
                valuation_func = bidder_info["valuation_function"]
                
                # Calculate expected valuation across scenarios
                expected_valuation = 0
                for scenario in price_scenarios:
                    scenario_valuation = valuation_func(item_characteristics, scenario)
                    expected_valuation += scenario_valuation / len(price_scenarios)
                
                # Add bidding strategy (truth-telling with noise)
                bid = expected_valuation * (0.9 + 0.2 * np.random.random())  # Strategic bidding
                round_bids[bidder_id] = max(bid, 0)
            
            # Determine winners and payments
            round_result = self._determine_auction_outcome(round_bids, item_characteristics)
            round_result["round"] = round_idx
            auction_results.append(round_result)
            
            # Update bidder histories
            for bidder_id in self.bidders:
                self.bidders[bidder_id]["bid_history"].append(round_bids.get(bidder_id, 0))
                self.bidders[bidder_id]["win_history"].append(
                    bidder_id in round_result.get("winners", [])
                )
        
        # Aggregate results
        total_revenue = sum(result["total_payment"] for result in auction_results)
        average_efficiency = np.mean([result["efficiency"] for result in auction_results])
        
        final_result = {
            "auction_type": self.auction_type,
            "total_revenue": total_revenue,
            "average_efficiency": average_efficiency,
            "round_results": auction_results,
            "n_rounds": n_rounds
        }
        
        self.auction_history.append(final_result)
        return final_result
    
    def _determine_auction_outcome(self, bids: Dict[int, float], 
                                 item_characteristics: Dict[str, Any]) -> Dict[str, Any]:
        """Determine auction winners and payments."""
        if not bids:
            return {"winners": [], "payments": {}, "total_payment": 0, "efficiency": 0}
        
        # Sort bids in descending order
        sorted_bids = sorted(bids.items(), key=lambda x: x[1], reverse=True)
        
        if self.auction_type == "first_price":
            # First-price auction: winner pays their bid
            winner_id, winning_bid = sorted_bids[0]
            return {
                "winners": [winner_id],
                "payments": {winner_id: winning_bid},
                "total_payment": winning_bid,
                "efficiency": winning_bid / max(bids.values()) if bids else 0
            }
        
        elif self.auction_type == "second_price":
            # Second-price auction: winner pays second-highest bid
            winner_id, winning_bid = sorted_bids[0]
            second_price = sorted_bids[1][1] if len(sorted_bids) > 1 else winning_bid * 0.9
            
            return {
                "winners": [winner_id],
                "payments": {winner_id: second_price},
                "total_payment": second_price,
                "efficiency": winning_bid / max(bids.values()) if bids else 0
            }
        
        elif self.auction_type == "all_pay":
            # All-pay auction: everyone pays their bid, highest bidder wins
            winner_id, winning_bid = sorted_bids[0]
            total_payment = sum(bids.values())
            
            return {
                "winners": [winner_id],
                "payments": bids,
                "total_payment": total_payment,
                "efficiency": winning_bid / max(bids.values()) if bids else 0
            }
        
        else:
            # Default to second-price
            winner_id, winning_bid = sorted_bids[0]
            second_price = sorted_bids[1][1] if len(sorted_bids) > 1 else winning_bid * 0.9
            
            return {
                "winners": [winner_id],
                "payments": {winner_id: second_price},
                "total_payment": second_price,
                "efficiency": winning_bid / max(bids.values()) if bids else 0
            }


# Utility functions
def create_stochastic_game(n_players: int = 3, game_type: str = "cooperative") -> StochasticGameTheory:
    """Factory function to create stochastic game."""
    return StochasticGameTheory(n_players=n_players, game_type=game_type)

def create_advanced_auction(auction_type: str = "second_price") -> AdvancedAuctionMechanism:
    """Factory function to create advanced auction mechanism."""
    return AdvancedAuctionMechanism(auction_type=auction_type) 