#!/usr/bin/env python

"""
Marathon of Twenty-One - Solution

This challenge involves playing multiple consecutive games of blackjack at each table.
The solution builds on the One Time Showdown implementation, but includes additional
strategy for managing bankroll across multiple games and adapting to observed dealer patterns.

The implementation maintains state across games and adjusts its strategy based on:
1. Observed dealer behavior patterns
2. Running count of cards seen
3. Current bankroll and optimal betting strategy
4. Prediction confidence and table-specific volatility
5. Adaptive strategy adjustments based on performance feedback
"""

import numpy as np
import os
import sys
from collections import deque

# Import our spy series prediction implementation
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.spy_series_solution import MyPlayer as SpyPlayer
from src.sherlocking_cards_solution import get_card_value_from_spy_value
from src.one_time_showdown_solution import MyPlayer as OneGamePlayer

class MyPlayer:
    """
    Player class for the Marathon of Twenty-One challenge.
    """
    
    def __init__(self, table_index):
        """
        Initialize the player with the table index.
        
        Args:
            table_index (int): Index of the table (0-4)
        """
        self.table_index = table_index
        self.spy_player = SpyPlayer(table_index)
        self.one_game_player = OneGamePlayer(table_index)
        
        # Game state tracking
        self.game_count = 0
        self.initial_bankroll = 250  # Starting bankroll
        self.bankroll = self.initial_bankroll
        self.cards_seen = []  # All cards seen so far
        
        # Card counting variables
        self.running_count = 0
        self.true_count = 0
        self.decks_remaining = 4  # Estimate of decks remaining
        
        # Performance tracking
        self.wins = 0
        self.losses = 0
        self.ties = 0
        self.win_streak = 0
        self.loss_streak = 0
        
        # Strategy parameters (can be adjusted per table)
        self.betting_unit = 5  # Base betting unit
        self.max_bet = 50     # Maximum bet
        self.min_bet = 5      # Minimum bet
        
        # Prediction tracking for confidence estimation
        self.prediction_history = []
        self.actual_values = []
        self.prediction_errors = deque(maxlen=10)  # Keep last 10 prediction errors
        
        # Table-specific strategy parameters
        self.strategy_params = {
            0: {"bet_scaling": 1.2, "surrender_threshold": 0.4, "prediction_confidence": 0.9},
            1: {"bet_scaling": 1.0, "surrender_threshold": 0.45, "prediction_confidence": 0.8},
            2: {"bet_scaling": 0.8, "surrender_threshold": 0.5, "prediction_confidence": 0.5},  # Low confidence in predictions
            3: {"bet_scaling": 0.7, "surrender_threshold": 0.55, "prediction_confidence": 0.7},
            4: {"bet_scaling": 0.6, "surrender_threshold": 0.6, "prediction_confidence": 0.75}
        }
        
        # Table-specific adjustments
        self._initialize_table_strategy()
    
    def _initialize_table_strategy(self):
        """Initialize table-specific strategy parameters."""
        if self.table_index == 0:
            # Table 0 - conservative approach with very predictable spy values
            self.betting_unit = 5
            self.max_bet = 30
            # High prediction confidence for Table 0
            self.base_prediction_confidence = 0.9
        elif self.table_index == 1:
            # Table 1 - moderate approach
            self.betting_unit = 5
            self.max_bet = 40
            self.base_prediction_confidence = 0.8
        elif self.table_index == 2:
            # Table 2 - aggressive approach but low prediction confidence (high variance table)
            self.betting_unit = 10
            self.max_bet = 50
            self.base_prediction_confidence = 0.5  # Lower confidence due to high MSE
        elif self.table_index == 3:
            # Table 3 - balanced approach
            self.betting_unit = 5
            self.max_bet = 30
            self.base_prediction_confidence = 0.7
        elif self.table_index == 4:
            # Table 4 - conservative approach (challenging table)
            self.betting_unit = 5
            self.max_bet = 25
            self.base_prediction_confidence = 0.75
    
    def get_bet_amount(self, player_cards, player_spy_history, 
                        dealer_cards, dealer_spy_history):
        """
        Determine the bet amount for the current game using Kelly Criterion.
        
        Args:
            player_cards: List of player's cards so far (empty at start of game)
            player_spy_history: List of player's spy values
            dealer_cards: List of dealer's visible cards (empty at start of game)
            dealer_spy_history: List of dealer's spy values
            
        Returns:
            float: Bet amount
        """
        # First game - use default starting bet
        if self.game_count == 0:
            self.game_count += 1
            return self.betting_unit
        
        # Update our estimate of the true count
        self._update_count()
        
        # Calculate prediction confidence
        prediction_confidence = self._get_prediction_confidence(player_spy_history)
        
        # Calculate win probability
        if self.true_count >= 2:
            # Positive count - favorable for player
            win_probability = 0.49  # Slightly under 0.5 (house edge still applies)
        elif self.true_count <= -2:
            # Negative count - unfavorable for player
            win_probability = 0.44
        else:
            # Neutral count
            win_probability = 0.46
            
        # Adjust win probability based on streak (psychological factor)
        if self.win_streak >= 3:
            win_probability = max(0.44, win_probability - 0.01)  # Slightly more conservative
        elif self.loss_streak >= 3:
            win_probability = min(0.51, win_probability + 0.01)  # Slightly more aggressive
            
        # Apply table-specific adjustments
        win_probability *= self.strategy_params[self.table_index]["bet_scaling"]
        
        # Kelly Criterion calculation
        # b: decimal odds (payoff to 1) - in blackjack typically 1:1
        # p: probability of winning
        # q: probability of losing (1-p)
        # Kelly fraction = (b*p - q) / b
        b = 1.0  # Even money payoff
        p = win_probability
        q = 1 - p
        kelly_fraction = max(0, (b*p - q) / b)
        
        # Apply confidence factor to Kelly
        adjusted_kelly = kelly_fraction * prediction_confidence
        
        # Never bet full Kelly (too risky)
        conservative_kelly = adjusted_kelly * 0.5  # Half-Kelly is more conservative
        
        # Set the bet amount based on conservative Kelly
        bet_amount = self.bankroll * conservative_kelly
        
        # Ensure our bet is within limits
        bet_amount = max(self.min_bet, min(bet_amount, self.max_bet))
        bet_amount = min(bet_amount, self.bankroll)  # Can't bet more than we have
        
        # Round to nearest 5
        bet_amount = round(bet_amount / 5) * 5
        if bet_amount == 0:
            bet_amount = self.min_bet
        
        self.game_count += 1
        return bet_amount
    
    def get_player_action(self, player_cards, player_spy_history, 
                        dealer_cards, dealer_spy_history,
                        player_total, dealer_total,
                        turn, dealer_bust_probability=None):
        """
        Determine the player's action based on the current game state.
        
        Args:
            player_cards: List of player's cards so far
            player_spy_history: List of player's spy values
            dealer_cards: List of dealer's visible cards
            dealer_spy_history: List of dealer's spy values
            player_total: Current total of player's cards
            dealer_total: Current total of dealer's visible cards
            turn: Whether it's the 'player' or 'dealer' turn
            dealer_bust_probability: Optional probability of dealer busting
            
        Returns:
            str: Action to take - 'hit'/'stand' for player, 'surrender'/'continue' for dealer
        """
        # Add seen cards to our tracking
        self._update_seen_cards(player_cards, dealer_cards)
        
        # Calculate prediction confidence
        prediction_confidence = self._get_prediction_confidence(player_spy_history)
        
        # Adjust the dealer bust probability using our confidence
        if dealer_bust_probability is not None:
            # Mix dealer bust probability with historical dealer bust rates based on confidence
            table_dealer_bust_rates = {
                0: 0.22,  # From Dealer's Doom analysis
                1: 0.18,
                2: 0.28,
                3: 0.20,
                4: 0.15
            }
            historical_bust_rate = table_dealer_bust_rates.get(self.table_index, 0.20)
            
            # If confidence is low, rely more on historical bust rate
            adjusted_bust_prob = (
                prediction_confidence * dealer_bust_probability + 
                (1 - prediction_confidence) * historical_bust_rate
            )
        else:
            adjusted_bust_prob = None
        
        # Use our one-game strategy as the base
        action = self.one_game_player.get_player_action(
            player_cards, player_spy_history,
            dealer_cards, dealer_spy_history,
            player_total, dealer_total,
            turn, adjusted_bust_prob
        )
        
        # Apply any marathon-specific adjustments
        if turn == 'player':
            # Player turn - considering card counting and game history
            action = self._adjust_player_action(
                action, player_total, dealer_total, self.true_count, prediction_confidence
            )
        else:  # turn == 'dealer'
            # Dealer turn - might adjust surrender strategy based on bankroll and confidence
            action = self._adjust_dealer_action(
                action, player_total, dealer_total, adjusted_bust_prob, prediction_confidence
            )
        
        return action
    
    def update_game_result(self, result, amount_won):
        """
        Update the player's state based on the game result.
        
        Args:
            result: Result of the game ('win', 'loss', or 'push')
            amount_won: Amount won (positive) or lost (negative)
        """
        # Update bankroll
        self.bankroll += amount_won
        
        # Update win/loss tracking
        if result == 'win':
            self.wins += 1
            self.win_streak += 1
            self.loss_streak = 0
        elif result == 'loss':
            self.losses += 1
            self.loss_streak += 1
            self.win_streak = 0
        else:  # result == 'push'
            self.ties += 1
            # Don't reset streaks on a push
        
        # Adjust strategy based on results if needed
        win_rate = self.wins / max(1, self.wins + self.losses)
        
        if self.game_count >= 10:
            # Dynamically adjust our strategy parameters based on performance
            params = self.strategy_params[self.table_index]
            
            if win_rate < 0.3:
                # We're losing a lot - become more conservative
                params["surrender_threshold"] = min(0.7, params["surrender_threshold"] + 0.05)
                params["bet_scaling"] = max(0.5, params["bet_scaling"] * 0.9)
            elif win_rate > 0.5:
                # We're winning a lot - can be slightly more aggressive
                params["surrender_threshold"] = max(0.3, params["surrender_threshold"] - 0.05)
                params["bet_scaling"] = min(1.5, params["bet_scaling"] * 1.1)
            
            # Reset if we're at extreme values to avoid getting stuck
            if self.bankroll < 50:
                # Getting low on funds, reset to be more conservative
                params["surrender_threshold"] = 0.6  # Higher threshold = more surrenders
                params["bet_scaling"] = 0.7  # Smaller bets
            elif self.bankroll > 400:
                # Doing very well, slightly more aggressive
                params["surrender_threshold"] = 0.4  # Lower threshold = fewer surrenders
    
    def _update_seen_cards(self, player_cards, dealer_cards):
        """
        Update the list of cards seen and card counting values.
        
        Args:
            player_cards: List of player's cards in this round
            dealer_cards: List of dealer's cards in this round
        """
        new_cards = player_cards + dealer_cards
        
        # Add to tracking only cards we haven't seen before
        for card in new_cards:
            if card not in self.cards_seen:
                self.cards_seen.append(card)
                
                # Update running count - High-Low system
                if card >= 10:  # 10, J, Q, K, A
                    self.running_count -= 1
                elif card <= 6:  # 2, 3, 4, 5, 6
                    self.running_count += 1
                # 7, 8, 9 are neutral (count doesn't change)
                
                # Estimate decks remaining
                self.decks_remaining = max(0.5, 4 - (len(self.cards_seen) / 52))
        
        # Update true count
        if self.decks_remaining > 0:
            self.true_count = self.running_count / self.decks_remaining
    
    def _update_count(self):
        """Update the true count estimate based on current observations."""
        # Rough estimate of how many cards we've seen
        cards_seen_estimate = min(208, len(self.cards_seen) + self.game_count * 5)
        self.decks_remaining = max(0.5, 4 - (cards_seen_estimate / 52))
        
        if self.decks_remaining > 0:
            self.true_count = self.running_count / self.decks_remaining
            
    def _get_prediction_confidence(self, spy_history):
        """
        Estimate confidence in our predictions based on recent accuracy.
        
        Args:
            spy_history: History of spy values to use for prediction
            
        Returns:
            float: Confidence score between 0 and 1
        """
        # If we have too little history, use table-specific base confidence
        if len(spy_history) < 5 or len(self.prediction_errors) < 3:
            return self.strategy_params[self.table_index]["prediction_confidence"]
            
        # Calculate confidence based on recent prediction errors
        if self.prediction_errors:
            mean_squared_error = np.mean(np.array(self.prediction_errors) ** 2)
            
            # Transform MSE into a confidence score
            # For each table, we have different expected MSE ranges
            table_mse_scales = {
                0: 1.0,      # Very low MSE
                1: 15.0,     # Moderate MSE
                2: 2000.0,   # Very high MSE
                3: 50.0,     # Moderate-high MSE
                4: 15.0      # Moderate MSE
            }
            
            # Scale MSE based on table characteristics
            scale = table_mse_scales.get(self.table_index, 20.0)
            
            # Convert to confidence score (0-1)
            confidence = max(0.1, min(0.95, 1.0 - (mean_squared_error / scale)))
            
            # For Table 2, apply additional dampening due to high volatility
            if self.table_index == 2:
                confidence = 0.5 + 0.5 * (confidence - 0.5)  # Reduce extremes
                
            return confidence
        else:
            return self.strategy_params[self.table_index]["prediction_confidence"]
    
    def _adjust_player_action(self, base_action, player_total, dealer_total, true_count, confidence):
        """
        Adjust the one-game player action based on count and confidence.
        
        Args:
            base_action: The action suggested by the one-game strategy
            player_total: Current total of player's cards
            dealer_total: Current total of dealer's visible cards
            true_count: The true count for card counting
            confidence: Confidence in our predictions
            
        Returns:
            str: Adjusted action ('hit' or 'stand')
        """
        # Critical decision zone
        if player_total >= 12 and player_total <= 16:
            # Dealer showing 7 or higher
            if dealer_total >= 7:
                # Positive count favors standing more often
                if true_count >= 3 and confidence > 0.6:
                    return 'stand'
                # Low confidence or negative count, follow basic strategy
                elif true_count <= -2 or confidence < 0.4:
                    return 'hit'
                else:
                    return base_action
            # Dealer showing 2-6 (bust cards)
            else:
                # Negative count favors hitting more often
                if true_count <= -3 and confidence > 0.6:
                    return 'hit'
                # Low confidence or positive count, follow basic strategy
                elif true_count >= 2 or confidence < 0.4:
                    return 'stand'
                else:
                    return base_action
        # Soft 17-18
        elif player_total == 17 or player_total == 18:
            # If ace is being counted as 11
            has_ace = any(card == 11 for card in self.one_game_player.player_cards)
            if has_ace and dealer_total >= 9:
                # Negative count favors hitting on soft 17-18 vs high dealer card
                if true_count <= -2 and confidence > 0.6:
                    return 'hit'
                else:
                    return base_action
        
        # For all other cases, use the base action
        return base_action
    
    def _adjust_dealer_action(self, base_action, player_total, dealer_total, 
                             dealer_bust_prob, confidence):
        """
        Adjust the dealer action (surrender decision) based on game state.
        
        Args:
            base_action: The action suggested by the one-game strategy
            player_total: Current total of player's cards
            dealer_total: Current total of dealer's visible cards
            dealer_bust_prob: Probability of dealer busting
            confidence: Confidence in our predictions
            
        Returns:
            str: Adjusted action ('surrender' or 'continue')
        """
        # Calculate the surrender threshold based on table-specific params and game state
        params = self.strategy_params[self.table_index]
        surrender_threshold = params["surrender_threshold"]
        
        # Adjust threshold based on bankroll situation
        bankroll_ratio = self.bankroll / self.initial_bankroll
        
        if bankroll_ratio < 0.3:
            # Very low bankroll - be more aggressive to try to recover
            surrender_threshold -= 0.15
        elif bankroll_ratio < 0.7:
            # Below starting bankroll - be slightly more aggressive
            surrender_threshold -= 0.05
        elif bankroll_ratio > 1.5:
            # Well above starting bankroll - be more conservative
            surrender_threshold += 0.05
            
        # If confidence is low, be more conservative with surrender
        if confidence < 0.5:
            surrender_threshold += 0.1
            
        # Make the surrender decision
        if player_total <= 11:
            return 'continue'  # Never surrender with 11 or less
        elif dealer_bust_prob is not None and dealer_bust_prob > 0.4:
            return 'continue'  # Continue if dealer has high bust probability
        elif player_total >= 17:
            return 'continue'  # Continue with 17 or higher
        else:
            # For 12-16, use our dynamic surrender threshold
            win_probability = self._estimate_win_probability(player_total, dealer_total)
            if win_probability < surrender_threshold:
                return 'surrender'
            else:
                return 'continue'
    
    def _estimate_win_probability(self, player_total, dealer_total):
        """
        Estimate the probability of winning with the current hand.
        
        Args:
            player_total: Current total of player's cards
            dealer_total: Current total of dealer's visible cards
            
        Returns:
            float: Estimated probability of winning
        """
        # Starting with basic probabilities
        if player_total >= 20:
            win_prob = 0.8
        elif player_total == 19:
            win_prob = 0.7
        elif player_total == 18:
            win_prob = 0.6
        elif player_total == 17:
            win_prob = 0.5
        elif player_total >= 13 and player_total <= 16:
            win_prob = 0.37
        else:  # 12 or less
            win_prob = 0.4
            
        # Adjust based on dealer's card
        if dealer_total >= 7:
            win_prob -= 0.1
        elif dealer_total <= 6:
            win_prob += 0.1
            
        # Adjust for true count
        win_prob += 0.02 * self.true_count
        
        # Ensure probability is within bounds
        return max(0.1, min(0.9, win_prob))

if __name__ == "__main__":
    print("Marathon of Twenty-One - Solution Test")
    print("=" * 50)
    
    # Test with a sample game sequence
    player = MyPlayer(0)
    
    # Sample first game state
    player_cards = [10, 6]  # Queen + 6 = 16
    player_spy_values = [10.0, 6.0]
    dealer_cards = [5]  # First card only
    dealer_spy_values = [5.0]
    player_total = sum(player_cards)
    dealer_total = sum(dealer_cards)
    
    # Get initial bet
    bet = player.get_bet_amount([], [], [], [])
    print(f"Initial bet: ${bet}")
    
    # Test player's turn
    action = player.get_player_action(
        player_cards, player_spy_values,
        dealer_cards, dealer_spy_values,
        player_total, dealer_total,
        'player'
    )
    print(f"Player's turn action: {action}")
    
    # Update game result
    player.update_game_result('win', 5)
    print(f"Bankroll after win: ${player.bankroll}")
    
    # Test next game bet
    next_bet = player.get_bet_amount([], [], [], [])
    print(f"Next bet: ${next_bet}")
    
    # Test dealer's turn in next game
    action = player.get_player_action(
        [8, 7], [8.0, 7.0],
        [10], [10.0],
        15, 10,
        'dealer'
    )
    print(f"Dealer's turn action: {action}") 