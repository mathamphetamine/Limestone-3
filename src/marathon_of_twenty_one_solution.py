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
"""

import numpy as np
import os
import sys

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
        
        # Strategy parameters (can be adjusted per table)
        self.betting_unit = 5  # Base betting unit
        self.max_bet = 50     # Maximum bet
        self.min_bet = 5      # Minimum bet
        
        # Table-specific adjustments
        self._initialize_table_strategy()
    
    def _initialize_table_strategy(self):
        """Initialize table-specific strategy parameters."""
        if self.table_index == 0:
            # Table 0 - conservative approach
            self.betting_unit = 5
            self.max_bet = 30
        elif self.table_index == 1:
            # Table 1 - moderate approach
            self.betting_unit = 5
            self.max_bet = 40
        elif self.table_index == 2:
            # Table 2 - aggressive approach (high variance table)
            self.betting_unit = 10
            self.max_bet = 50
        elif self.table_index == 3:
            # Table 3 - balanced approach
            self.betting_unit = 5
            self.max_bet = 30
        elif self.table_index == 4:
            # Table 4 - conservative approach (challenging table)
            self.betting_unit = 5
            self.max_bet = 25
    
    def get_bet_amount(self, player_cards, player_spy_history, 
                        dealer_cards, dealer_spy_history):
        """
        Determine the bet amount for the current game.
        
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
        
        # Base bet on true count
        if self.true_count >= 2:
            # Positive count - favorable for player
            bet_amount = min(self.betting_unit * max(1, self.true_count), self.max_bet)
        elif self.true_count <= -2:
            # Negative count - unfavorable for player
            bet_amount = self.min_bet
        else:
            # Neutral count
            bet_amount = self.betting_unit
        
        # Adjust based on bankroll
        bankroll_ratio = self.bankroll / self.initial_bankroll
        if bankroll_ratio < 0.5:
            # We're down significantly - be more conservative
            bet_amount = max(self.min_bet, bet_amount * 0.75)
        elif bankroll_ratio > 1.5:
            # We're up significantly - can be more aggressive
            bet_amount = min(self.max_bet, bet_amount * 1.25)
        
        # Ensure our bet is within our limits
        bet_amount = max(self.min_bet, min(bet_amount, self.max_bet))
        bet_amount = min(bet_amount, self.bankroll)  # Can't bet more than we have
        
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
        
        # Use our one-game strategy as the base
        action = self.one_game_player.get_player_action(
            player_cards, player_spy_history,
            dealer_cards, dealer_spy_history,
            player_total, dealer_total,
            turn, dealer_bust_probability
        )
        
        # Apply any marathon-specific adjustments
        if turn == 'player':
            # Player turn - considering card counting and game history
            action = self._adjust_player_action(
                action, player_total, dealer_total, self.true_count
            )
        else:  # turn == 'dealer'
            # Dealer turn - might adjust surrender strategy based on bankroll
            action = self._adjust_dealer_action(
                action, player_total, dealer_total, dealer_bust_probability
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
        elif result == 'loss':
            self.losses += 1
        else:  # result == 'push'
            self.ties += 1
        
        # Adjust strategy based on results if needed
        win_rate = self.wins / max(1, self.wins + self.losses)
        
        if self.game_count >= 10 and win_rate < 0.3:
            # We're losing a lot - adjust to be more conservative
            self.max_bet = max(self.min_bet, self.max_bet * 0.9)
            self.betting_unit = max(self.min_bet, self.betting_unit * 0.9)
        elif self.game_count >= 10 and win_rate > 0.5:
            # We're winning a lot - can be slightly more aggressive
            self.max_bet = min(50, self.max_bet * 1.1)
            self.betting_unit = min(10, self.betting_unit * 1.1)
    
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
    
    def _adjust_player_action(self, base_action, player_total, dealer_total, true_count):
        """
        Adjust player action based on card counting and game history.
        
        Args:
            base_action: Action from one-game strategy
            player_total: Current total of player's cards
            dealer_total: Current total of dealer's visible cards
            true_count: Current true count from card counting
            
        Returns:
            str: Adjusted action ('hit' or 'stand')
        """
        # If the count is highly positive, we're more likely to see low cards
        if true_count >= 3 and player_total >= 12 and player_total <= 16:
            # More conservative - stand on 12-16 against dealer 2-6
            if dealer_total >= 2 and dealer_total <= 6:
                return 'stand'
        
        # If the count is highly negative, we're more likely to see high cards
        if true_count <= -3 and player_total >= 12 and player_total <= 16:
            # More aggressive - hit on 12-16 unless against dealer 6
            if dealer_total != 6:
                return 'hit'
        
        # If we're doing well financially, we can be more aggressive
        if self.bankroll > self.initial_bankroll * 1.5 and player_total >= 17 and player_total <= 18:
            # If dealer shows a strong card, consider hitting on 17-18
            if dealer_total >= 9:
                return 'hit'
        
        return base_action
    
    def _adjust_dealer_action(self, base_action, player_total, dealer_total, dealer_bust_prob):
        """
        Adjust dealer action based on bankroll and game history.
        
        Args:
            base_action: Action from one-game strategy
            player_total: Current total of player's cards
            dealer_total: Current total of dealer's visible cards
            dealer_bust_prob: Probability of dealer busting
            
        Returns:
            str: Adjusted action ('surrender' or 'continue')
        """
        # If we're doing poorly, be more conservative with surrenders
        if self.bankroll < self.initial_bankroll * 0.7:
            # We need to win back money, so continue more often
            if player_total >= 14 and dealer_bust_prob >= 0.25:
                return 'continue'
        
        # If we're doing very well, can afford to be more strategic with surrenders
        if self.bankroll > self.initial_bankroll * 1.3:
            # Lock in half our bet more often when we have a weak hand
            if player_total <= 14 and dealer_total >= 9:
                return 'surrender'
        
        return base_action

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