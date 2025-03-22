#!/usr/bin/env python

"""
Sicilian Synergy - Solution

This challenge involves playing blackjack simultaneously across 3 different tables,
with the ability to observe spy values across tables before making decisions.

The implementation coordinates information from all available tables to make
more informed decisions, and uses a shared bankroll strategy to optimize
overall performance.
"""

import numpy as np
import os
import sys

# Import our existing solutions
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.marathon_of_twenty_one_solution import MyPlayer as MarathonPlayer
from src.one_time_showdown_solution import MyPlayer as OneGamePlayer
from src.spy_series_solution import MyPlayer as SpyPlayer
from src.sherlocking_cards_solution import get_card_value_from_spy_value

class PlayerMulti:
    """
    Player class for the Sicilian Synergy challenge.
    Coordinates play across 3 tables.
    """
    
    def __init__(self, table_indices=[0, 1, 2]):
        """
        Initialize the player with the indices of the tables to be played simultaneously.
        
        Args:
            table_indices: List of table indices to play (default: [0, 1, 2])
        """
        self.table_indices = table_indices
        self.num_tables = len(table_indices)
        
        # Create individual players for each table
        self.players = {
            table_idx: MarathonPlayer(table_idx) 
            for table_idx in table_indices
        }
        
        # Create spy prediction models for all tables
        self.spy_players = {
            table_idx: SpyPlayer(table_idx)
            for table_idx in range(5)  # Initialize all 5 tables
        }
        
        # Shared bankroll for all tables
        self.initial_bankroll = 250 * self.num_tables  # Starting bankroll
        self.bankroll = self.initial_bankroll
        
        # Performance tracking
        self.wins = {table_idx: 0 for table_idx in table_indices}
        self.losses = {table_idx: 0 for table_idx in table_indices}
        self.ties = {table_idx: 0 for table_idx in table_indices}
        
        # Game state tracking
        self.game_count = 0
        self.cards_seen = []  # All cards seen across all tables
        
        # Card counting variables
        self.running_count = 0
        self.true_count = 0
        self.decks_remaining = 4 * self.num_tables  # Estimate of decks remaining
        
        # Table reputation system (which tables perform better)
        self.table_reputation = {table_idx: 0.0 for table_idx in table_indices}
        
        # Cross-table information sharing
        self.table_spy_histories = {
            table_idx: {'player': [], 'dealer': []} 
            for table_idx in range(5)  # Keep track of all tables
        }
    
    def get_bet_amount(self, table_index, player_cards, player_spy_history, 
                        dealer_cards, dealer_spy_history):
        """
        Determine the bet amount for the current game on a specific table.
        
        Args:
            table_index: Index of the current table
            player_cards: List of player's cards so far (empty at start of game)
            player_spy_history: List of player's spy values
            dealer_cards: List of dealer's visible cards (empty at start of game)
            dealer_spy_history: List of dealer's spy values
            
        Returns:
            float: Bet amount
        """
        # Update spy history for this table
        self._update_spy_history(table_index, player_spy_history, dealer_spy_history)
        
        # First game - allocate bankroll based on table reputation
        if self.game_count == 0:
            self.game_count += 1
            # Initial bets are equal across tables
            return 5  # Minimum bet to start
        
        # Update card counting
        self._update_count()
        
        # Base bet on true count
        if self.true_count >= 2:
            # Positive count - favorable for player
            base_bet = 5 * max(1, self.true_count)
        elif self.true_count <= -2:
            # Negative count - unfavorable for player
            base_bet = 5  # Minimum bet
        else:
            # Neutral count
            base_bet = 10  # Default bet
        
        # Adjust based on table reputation
        table_factor = 1.0 + self.table_reputation[table_index]
        bet_amount = base_bet * table_factor
        
        # Adjust based on bankroll
        bankroll_ratio = self.bankroll / self.initial_bankroll
        if bankroll_ratio < 0.5:
            # We're down significantly - be more conservative
            bet_amount = bet_amount * 0.75
        elif bankroll_ratio > 1.5:
            # We're up significantly - can be more aggressive
            bet_amount = bet_amount * 1.25
        
        # Cross-table prediction adjustment
        player_advantage = self._calculate_player_advantage(
            table_index, player_spy_history, dealer_spy_history
        )
        if player_advantage > 0.1:
            # Increase bet if we have a predicted advantage on this table
            bet_amount = bet_amount * (1 + player_advantage)
        elif player_advantage < -0.1:
            # Decrease bet if disadvantage predicted
            bet_amount = max(5, bet_amount * 0.5)
        
        # Ensure our bet is within our limits
        bet_amount = max(5, min(bet_amount, 50))  # Min 5, max 50
        
        # Ensure we don't bet more than our bankroll and distribute reasonably
        max_per_table = self.bankroll / self.num_tables * 2  # Allow up to 2x per table
        bet_amount = min(bet_amount, max_per_table)
        
        return bet_amount
    
    def get_player_action(self, table_index, player_cards, player_spy_history, 
                        dealer_cards, dealer_spy_history,
                        player_total, dealer_total,
                        turn, dealer_bust_probability=None):
        """
        Determine the player's action based on the current game state.
        
        Args:
            table_index: Index of the current table
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
        # Update spy history for this table
        self._update_spy_history(table_index, player_spy_history, dealer_spy_history)
        
        # Add seen cards to our tracking
        self._update_seen_cards(player_cards, dealer_cards)
        
        # Get base action from the individual table player
        player = self.players[table_index]
        base_action = player.get_player_action(
            player_cards, player_spy_history,
            dealer_cards, dealer_spy_history,
            player_total, dealer_total,
            turn, dealer_bust_probability
        )
        
        # Apply cross-table knowledge to refine our action
        if turn == 'player':
            # Player turn - use cross-table information
            action = self._adjust_player_action_with_cross_table_info(
                table_index, base_action,
                player_total, dealer_total,
                player_spy_history, dealer_spy_history
            )
        else:  # turn == 'dealer'
            # Dealer turn - might adjust surrender strategy based on cross-table
            action = self._adjust_dealer_action_with_cross_table_info(
                table_index, base_action,
                player_total, dealer_total,
                dealer_bust_probability
            )
        
        return action
    
    def update_game_result(self, table_index, result, amount_won):
        """
        Update the player's state based on the game result.
        
        Args:
            table_index: Index of the table the result is for
            result: Result of the game ('win', 'loss', or 'push')
            amount_won: Amount won (positive) or lost (negative)
        """
        # Update bankroll
        self.bankroll += amount_won
        
        # Update win/loss tracking for this table
        if result == 'win':
            self.wins[table_index] += 1
        elif result == 'loss':
            self.losses[table_index] += 1
        else:  # result == 'push'
            self.ties[table_index] += 1
        
        # Update individual table player
        player = self.players[table_index]
        player.update_game_result(result, amount_won)
        
        # Update table reputation based on results
        win_rate = self.wins[table_index] / max(1, self.wins[table_index] + self.losses[table_index])
        
        # Adjust reputation based on performance
        if result == 'win':
            self.table_reputation[table_index] += 0.05
        elif result == 'loss':
            self.table_reputation[table_index] -= 0.03
        
        # Normalize reputation to prevent extremes
        self.table_reputation[table_index] = max(-0.5, min(0.5, self.table_reputation[table_index]))
    
    def _update_spy_history(self, table_index, player_spy_history, dealer_spy_history):
        """
        Update the recorded spy histories for a table.
        
        Args:
            table_index: Index of the table to update
            player_spy_history: List of player's spy values
            dealer_spy_history: List of dealer's spy values
        """
        if len(player_spy_history) > len(self.table_spy_histories[table_index]['player']):
            self.table_spy_histories[table_index]['player'] = player_spy_history.copy()
        
        if len(dealer_spy_history) > len(self.table_spy_histories[table_index]['dealer']):
            self.table_spy_histories[table_index]['dealer'] = dealer_spy_history.copy()
    
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
                total_cards = 52 * 4 * self.num_tables  # 4 decks per table
                cards_used = len(self.cards_seen)
                self.decks_remaining = max(0.5, (total_cards - cards_used) / 52)
        
        # Update true count
        if self.decks_remaining > 0:
            self.true_count = self.running_count / self.decks_remaining
    
    def _update_count(self):
        """Update the true count estimate based on current observations."""
        # Ensure count is up to date
        if self.decks_remaining > 0:
            self.true_count = self.running_count / self.decks_remaining
    
    def _calculate_player_advantage(self, table_index, player_spy_history, dealer_spy_history):
        """
        Calculate player advantage using cross-table information.
        
        Args:
            table_index: Index of the current table
            player_spy_history: List of player's spy values
            dealer_spy_history: List of dealer's spy values
            
        Returns:
            float: Estimated player advantage (-1.0 to 1.0)
        """
        advantage = 0.0
        
        # Check if we have enough history for predictions
        if len(player_spy_history) >= 5 and len(dealer_spy_history) >= 5:
            # Get predictions for next player and dealer cards
            player_spy = self.spy_players[table_index].get_player_spy_prediction(
                np.array(player_spy_history[-5:])
            )
            dealer_spy = self.spy_players[table_index].get_dealer_spy_prediction(
                np.array(dealer_spy_history[-5:])
            )
            
            # Convert to card values
            player_card = get_card_value_from_spy_value(player_spy)
            dealer_card = get_card_value_from_spy_value(dealer_spy)
            
            # Compare predicted values
            if player_card > dealer_card:
                advantage += 0.1
            elif dealer_card > player_card:
                advantage -= 0.1
        
        # Use cross-table information
        for other_table in self.table_indices:
            if other_table != table_index:
                # If the other table is doing very well, it might indicate
                # favorable overall conditions
                win_rate = self.wins[other_table] / max(1, self.wins[other_table] + self.losses[other_table])
                if win_rate > 0.6:
                    advantage += 0.05
                elif win_rate < 0.3:
                    advantage -= 0.05
        
        # Consider card counting
        if self.true_count >= 2:
            advantage += 0.1 * (self.true_count - 1)
        elif self.true_count <= -2:
            advantage -= 0.1 * abs(self.true_count + 1)
        
        return advantage
    
    def _check_for_spy_patterns(self, table_index, player_spy_history, dealer_spy_history):
        """
        Check for patterns in spy values across tables that might indicate
        favorable conditions.
        
        Args:
            table_index: Index of the current table
            player_spy_history: List of player's spy values
            dealer_spy_history: List of dealer's spy values
            
        Returns:
            dict: Dictionary of detected patterns and their strengths
        """
        patterns = {}
        
        # Check for alternating pattern in dealer values (common in table 0)
        if len(dealer_spy_history) >= 4:
            alternating = True
            for i in range(len(dealer_spy_history) - 3, len(dealer_spy_history) - 1):
                if (dealer_spy_history[i] > 0 and dealer_spy_history[i+1] > 0) or \
                   (dealer_spy_history[i] < 0 and dealer_spy_history[i+1] < 0):
                    alternating = False
                    break
            
            if alternating:
                patterns['dealer_alternating'] = True
        
        # Look for correlations between tables
        for other_table in self.table_indices:
            if other_table != table_index:
                other_player_spy = self.table_spy_histories[other_table]['player']
                other_dealer_spy = self.table_spy_histories[other_table]['dealer']
                
                # Check if we have enough data
                if len(other_player_spy) >= 5 and len(other_dealer_spy) >= 5:
                    # Simple correlation: check if recent spy trends match
                    if (player_spy_history[-1] > player_spy_history[-2] and 
                        other_player_spy[-1] > other_player_spy[-2]):
                        patterns[f'player_correlation_table_{other_table}'] = True
                    
                    if (dealer_spy_history[-1] > dealer_spy_history[-2] and 
                        other_dealer_spy[-1] > other_dealer_spy[-2]):
                        patterns[f'dealer_correlation_table_{other_table}'] = True
        
        return patterns
    
    def _adjust_player_action_with_cross_table_info(self, table_index, base_action,
                                                  player_total, dealer_total,
                                                  player_spy_history, dealer_spy_history):
        """
        Adjust player action using cross-table information.
        
        Args:
            table_index: Index of the current table
            base_action: Base action from individual table player
            player_total: Current total of player's cards
            dealer_total: Current total of dealer's visible cards
            player_spy_history: List of player's spy values
            dealer_spy_history: List of dealer's spy values
            
        Returns:
            str: Adjusted action ('hit' or 'stand')
        """
        # Check for patterns across tables
        patterns = self._check_for_spy_patterns(
            table_index, player_spy_history, dealer_spy_history
        )
        
        # If we detect strong dealer alternating pattern in table 0,
        # we can predict next dealer card with high confidence
        if table_index == 0 and patterns.get('dealer_alternating'):
            next_dealer_spy = -dealer_spy_history[-1]
            next_dealer_card = get_card_value_from_spy_value(next_dealer_spy)
            
            # If dealer's predicted next card is high (risk of strong dealer hand)
            if next_dealer_card >= 9 and player_total >= 17:
                # More aggressive with a very strong hand
                return base_action
            elif next_dealer_card >= 9 and player_total < 17:
                # More aggressive with weaker hand against strong dealer
                return 'hit'
            elif next_dealer_card <= 6 and player_total >= 12:
                # More conservative with decent hand against weak dealer
                return 'stand'
        
        # Use correlations between tables to inform decisions
        correlations_detected = sum(1 for k, v in patterns.items() if 'correlation' in k and v)
        if correlations_detected >= 2:
            # Strong evidence of correlations across tables
            player_advantage = self._calculate_player_advantage(
                table_index, player_spy_history, dealer_spy_history
            )
            
            if player_advantage > 0.2 and player_total >= 16:
                # If advantage detected and reasonable hand, stand
                return 'stand'
            elif player_advantage < -0.2 and player_total < 17:
                # If disadvantage detected, be more aggressive
                return 'hit'
        
        # If true count is very high and we have a marginal hand,
        # be more likely to stand (expecting low cards)
        if self.true_count >= 3 and 12 <= player_total <= 16:
            return 'stand'
        
        # If true count is very negative and we have a marginal hand,
        # be more likely to hit (expecting high cards)
        if self.true_count <= -3 and 12 <= player_total <= 16:
            return 'hit'
        
        return base_action
    
    def _adjust_dealer_action_with_cross_table_info(self, table_index, base_action,
                                                 player_total, dealer_total,
                                                 dealer_bust_probability=None):
        """
        Adjust dealer action using cross-table information.
        
        Args:
            table_index: Index of the current table
            base_action: Base action from individual table player
            player_total: Current total of player's cards
            dealer_total: Current total of dealer's visible cards
            dealer_bust_probability: Probability of dealer busting
            
        Returns:
            str: Adjusted action ('surrender' or 'continue')
        """
        # If we're down in bankroll, take fewer risks
        if self.bankroll < self.initial_bankroll * 0.8:
            # If player has a reasonable hand and dealer not too strong
            if player_total >= 15 and dealer_total <= 7:
                return 'continue'
        
        # If we're up in bankroll, can be more strategic
        if self.bankroll > self.initial_bankroll * 1.2:
            # If player has weak hand and dealer strong upcard
            if player_total <= 14 and dealer_total >= 9:
                return 'surrender'
        
        # Consider card counting for surrender decisions
        if self.true_count >= 3:
            # High count means more low cards remaining
            if dealer_total <= 6 and player_total >= 12:
                # Dealer might bust with low cards
                return 'continue'
        elif self.true_count <= -3:
            # Low count means more high cards remaining
            if dealer_total <= 5 and player_total <= 16:
                # Dealer likely to get good cards
                return 'surrender'
        
        # Check for table correlations
        # If other tables are showing patterns that suggest a disadvantage,
        # be more likely to surrender
        disadvantaged_tables = 0
        for other_table in self.table_indices:
            if other_table != table_index and self.table_reputation[other_table] < -0.2:
                disadvantaged_tables += 1
        
        if disadvantaged_tables >= 2:
            # If most tables showing disadvantage, be more cautious
            if player_total <= 16:
                return 'surrender'
        
        return base_action

if __name__ == "__main__":
    print("Sicilian Synergy - Solution Test")
    print("=" * 50)
    
    # Test with a sample multi-table setup
    player = PlayerMulti([0, 1, 2])
    
    # Sample game state for Table 0
    table_index = 0
    player_cards = [10, 6]  # Queen + 6 = 16
    player_spy_values = [10.0, 6.0]
    dealer_cards = [5]  # First card only
    dealer_spy_values = [5.0]
    player_total = sum(player_cards)
    dealer_total = sum(dealer_cards)
    
    # Get initial bet
    bet = player.get_bet_amount(table_index, [], [], [], [])
    print(f"Initial bet for Table {table_index}: ${bet}")
    
    # Test player's turn
    action = player.get_player_action(
        table_index,
        player_cards, player_spy_values,
        dealer_cards, dealer_spy_values,
        player_total, dealer_total,
        'player'
    )
    print(f"Player's turn action for Table {table_index}: {action}")
    
    # Update game result
    player.update_game_result(table_index, 'win', 5)
    print(f"Bankroll after win on Table {table_index}: ${player.bankroll}")
    print(f"Table reputation for Table {table_index}: {player.table_reputation[table_index]}")
    
    # Test with Table 1
    table_index = 1
    bet = player.get_bet_amount(table_index, [], [], [], [])
    print(f"Bet for Table {table_index}: ${bet}")
    
    # Test dealer's turn
    action = player.get_player_action(
        table_index,
        [8, 7], [8.0, 7.0],
        [10], [10.0],
        15, 10,
        'dealer'
    )
    print(f"Dealer's turn action for Table {table_index}: {action}") 