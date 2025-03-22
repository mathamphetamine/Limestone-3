#!/usr/bin/env python

"""
One Time Showdown - Solution

This challenge involves playing a single game of blackjack at each table.
The solution uses the spy series prediction model from Part 3 and the
card-value function from Part 2 to make optimal decisions.

The implementation decides whether to hit or stand during the player's turn,
and whether to surrender or continue during the dealer's turn.
"""

import numpy as np
import os
import sys

# Import our spy series prediction implementation
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.spy_series_solution import MyPlayer as SpyPlayer
from src.sherlocking_cards_solution import get_card_value_from_spy_value

class MyPlayer:
    """
    Player class for the One Time Showdown challenge.
    """
    
    def __init__(self, table_index):
        """
        Initialize the player with the table index.
        
        Args:
            table_index (int): Index of the table (0-4)
        """
        self.table_index = table_index
        self.spy_player = SpyPlayer(table_index)
        
        # Store information about the current game
        self.player_cards = []
        self.player_spy_values = []
        self.dealer_cards = []
        self.dealer_spy_values = []
        
        # Initialize statistical variables
        self.dealer_bust_probs = {
            0: 0.22,  # These values can be calculated using dealers_doom_solution.py
            1: 0.18,  # But for now, we'll use reasonable estimates
            2: 0.28,
            3: 0.20,
            4: 0.15
        }
    
    def get_player_action(self, 
                        player_cards, player_spy_history, 
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
        # Update our game state
        self.player_cards = player_cards.copy()
        self.player_spy_values = player_spy_history.copy()
        self.dealer_cards = dealer_cards.copy()
        self.dealer_spy_values = dealer_spy_history.copy()
        
        # Use the provided dealer bust probability if available, otherwise use our estimate
        if dealer_bust_probability is not None:
            bust_prob = dealer_bust_probability
        else:
            bust_prob = self.dealer_bust_probs.get(self.table_index, 0.2)
        
        # Player's turn - decide whether to hit or stand
        if turn == 'player':
            return self._get_player_turn_action(player_total, dealer_total)
        
        # Dealer's turn - decide whether to surrender or continue
        else:  # turn == 'dealer'
            return self._get_dealer_turn_action(player_total, dealer_total, bust_prob)
    
    def _get_player_turn_action(self, player_total, dealer_total):
        """
        Determine action during player's turn: hit or stand.
        
        Args:
            player_total: Current total of player's cards
            dealer_total: Current total of dealer's visible cards
            
        Returns:
            str: 'hit' or 'stand'
        """
        # If we already have 21, always stand
        if player_total == 21:
            return 'stand'
        
        # If we're at 20, almost always stand (hit only in very specific situations)
        if player_total == 20:
            return 'stand'
        
        # Basic strategy: hit on 16 or below, except against dealer 6 or lower
        if player_total <= 11:
            return 'hit'  # Always hit on 11 or below
        
        # Use our prediction model to estimate the next player card
        next_player_spy = None
        if len(self.player_spy_values) >= 5:
            next_player_spy = self.spy_player.get_player_spy_prediction(
                np.array(self.player_spy_values[-5:])
            )
            next_player_card = get_card_value_from_spy_value(next_player_spy)
            
            # If hitting would bust us and our total is decent, stand
            if player_total + next_player_card > 21 and player_total >= 14:
                return 'stand'
        
        # Strategy based on player's total and dealer's upcard
        if player_total >= 17:
            return 'stand'  # Generally stand on 17+
        
        if player_total >= 13 and dealer_total <= 6:
            return 'stand'  # Stand against dealer 2-6 with 13+
        
        if player_total == 12 and 4 <= dealer_total <= 6:
            return 'stand'  # Stand with 12 against 4-6
            
        # Default to hitting in uncertain situations
        return 'hit'
    
    def _get_dealer_turn_action(self, player_total, dealer_total, dealer_bust_prob):
        """
        Determine action during dealer's turn: surrender or continue.
        
        Args:
            player_total: Current total of player's cards
            dealer_total: Current total of dealer's visible cards
            dealer_bust_prob: Probability of dealer busting
            
        Returns:
            str: 'surrender' or 'continue'
        """
        # We can only surrender if dealer's total <= 16
        if dealer_total > 16:
            return 'continue'
        
        # If we have a low score, surrender is often better
        if player_total < 13:
            return 'surrender'
        
        # If dealer has a good chance of busting, continue
        if dealer_bust_prob > 0.3:
            return 'continue'
        
        # Calculate our win probability based on the current state
        win_prob = self._calculate_win_probability(player_total, dealer_total)
        
        # Surrender if our win probability is below 0.33
        # (Expected value of surrender is 0.5, while continue is win_prob)
        if win_prob < 0.33:
            return 'surrender'
        
        return 'continue'
    
    def _calculate_win_probability(self, player_total, dealer_total):
        """
        Calculate probability of winning given current totals.
        
        Args:
            player_total: Current total of player's cards
            dealer_total: Current total of dealer's visible cards
            
        Returns:
            float: Probability of winning (0.0 to 1.0)
        """
        # Estimate dealer's final total distribution
        dealer_outcomes = self._simulate_dealer_outcomes(dealer_total)
        
        # Calculate probability of each outcome
        win_prob = 0.0
        for outcome, prob in dealer_outcomes.items():
            # Dealer busts
            if outcome > 21:
                win_prob += prob
            # Dealer's total is less than player's
            elif outcome < player_total:
                win_prob += prob
            # Push (tie) case
            elif outcome == player_total:
                # No change in win probability
                pass
            # Dealer wins
            else:
                # No change in win probability
                pass
        
        return win_prob
    
    def _simulate_dealer_outcomes(self, dealer_total):
        """
        Simulate possible dealer outcomes to estimate probability distribution.
        
        Args:
            dealer_total: Current total of dealer's visible cards
            
        Returns:
            dict: Mapping from dealer final total to probability
        """
        # Simple model: assume dealer will end up with this distribution
        outcomes = {}
        
        # If dealer has 16 or less, they must hit
        if dealer_total <= 16:
            # Predict next dealer card using spy values
            next_dealer_spy = None
            if len(self.dealer_spy_values) >= 5:
                next_dealer_spy = self.spy_player.get_dealer_spy_prediction(
                    np.array(self.dealer_spy_values[-5:])
                )
                next_dealer_card = get_card_value_from_spy_value(next_dealer_spy)
                
                # Calculate new total
                new_total = dealer_total + next_dealer_card
                
                if new_total > 21:
                    # Dealer busts
                    outcomes[new_total] = 1.0
                elif new_total >= 17:
                    # Dealer stands
                    outcomes[new_total] = 1.0
                else:
                    # Need another card - simplify by using average distribution
                    outcomes[17] = 0.2  # Dealer gets exactly 17
                    outcomes[18] = 0.15  # Dealer gets 18
                    outcomes[19] = 0.15  # Dealer gets 19
                    outcomes[20] = 0.15  # Dealer gets 20
                    outcomes[21] = 0.15  # Dealer gets 21
                    outcomes[22] = 0.2  # Dealer busts
            else:
                # Not enough spy history, use a generic distribution
                outcomes[17] = 0.2  # Dealer gets exactly 17
                outcomes[18] = 0.15  # Dealer gets 18
                outcomes[19] = 0.15  # Dealer gets 19
                outcomes[20] = 0.15  # Dealer gets 20
                outcomes[21] = 0.15  # Dealer gets 21
                outcomes[22] = 0.2  # Dealer busts
        else:
            # Dealer already standing
            outcomes[dealer_total] = 1.0
        
        return outcomes

if __name__ == "__main__":
    print("One Time Showdown - Solution Test")
    print("=" * 50)
    
    # Test with a sample game state
    player = MyPlayer(0)
    
    # Sample game state
    player_cards = [10, 6]  # Queen + 6 = 16
    player_spy_values = [10.0, 6.0]
    dealer_cards = [5]  # First card only
    dealer_spy_values = [5.0]
    player_total = sum(player_cards)
    dealer_total = sum(dealer_cards)
    
    # Test player's turn
    action = player.get_player_action(
        player_cards, player_spy_values,
        dealer_cards, dealer_spy_values,
        player_total, dealer_total,
        'player'
    )
    print(f"Player's turn action: {action}")
    
    # Test dealer's turn
    action = player.get_player_action(
        player_cards, player_spy_values,
        dealer_cards, dealer_spy_values,
        player_total, dealer_total,
        'dealer'
    )
    print(f"Dealer's turn action: {action}") 