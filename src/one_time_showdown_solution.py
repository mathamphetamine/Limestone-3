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
        
        # Table-specific strategy adjustments
        self.table_strategy = {
            0: {"stand_soft_17": True, "dealer_aggressive": False},
            1: {"stand_soft_17": True, "dealer_aggressive": False},
            2: {"stand_soft_17": False, "dealer_aggressive": True},
            3: {"stand_soft_17": True, "dealer_aggressive": False},
            4: {"stand_soft_17": True, "dealer_aggressive": True}
        }
        
        # Initialize statistical variables from dealers_doom_solution analysis
        self.dealer_bust_probs = {
            0: 0.22,
            1: 0.18,
            2: 0.28,
            3: 0.20,
            4: 0.15
        }
        
        # Expected dealer final totals from dealers_doom_solution
        self.dealer_expected_totals = {
            0: 18.4,
            1: 18.7,
            2: 18.1,
            3: 18.5,
            4: 19.1
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
        
        # Check if player has a soft hand (ace counted as 11)
        has_ace = 11 in player_cards
        
        # Player's turn - decide whether to hit or stand
        if turn == 'player':
            return self._get_player_turn_action(player_total, dealer_total, has_ace)
        
        # Dealer's turn - decide whether to surrender or continue
        else:  # turn == 'dealer'
            return self._get_dealer_turn_action(player_total, dealer_total, bust_prob, has_ace)
    
    def _get_player_turn_action(self, player_total, dealer_total, has_ace):
        """
        Determine action during player's turn: hit or stand.
        
        Args:
            player_total: Current total of player's cards
            dealer_total: Current total of dealer's visible cards
            has_ace: Whether the player has an ace counted as 11
            
        Returns:
            str: 'hit' or 'stand'
        """
        # If we already have 21, always stand
        if player_total == 21:
            return 'stand'
        
        # Estimate next card based on spy prediction
        predicted_card = self._predict_next_card(self.player_spy_values, is_player=True)
        
        # Handle soft hands (when an ace is counted as 11)
        if has_ace:
            # Soft totals
            if player_total >= 19:
                return 'stand'  # Always stand on soft 19+
            elif player_total == 18:
                # Stand on soft 18 against dealer 2-8, hit against 9-A
                if 2 <= dealer_total <= 8:
                    return 'stand'
                else:
                    # If hitting is unlikely to bust us (because ace will convert to 1), hit
                    return 'hit'
            else:  # Soft 17 or lower
                # Hit on soft 17 or lower
                return 'hit'
        
        # Hard totals (no ace counted as 11)
        
        # Always hit on 11 or lower - no risk of busting
        if player_total <= 11:
            return 'hit'
            
        # Always stand on hard 17 or higher
        if player_total >= 17:
            return 'stand'
            
        # Critical decision area: 12-16
        
        # Check if next hit would bust based on prediction
        if predicted_card and player_total + predicted_card > 21:
            # Next hit would likely bust us
            # But if dealer is showing 7+ and we have poor hand, may still need to hit
            if player_total <= 13 and dealer_total >= 7:
                return 'hit'  # Take the risk
            return 'stand'
            
        # Decision making for 12-16 based on dealer upcard
        if player_total == 16:
            # Special adjustment for Table 2 which has higher dealer bust probability
            if self.table_index == 2 and dealer_total <= 6:
                return 'stand'
            return 'stand' if dealer_total <= 6 else 'hit'
        elif player_total == 15:
            return 'stand' if dealer_total <= 6 else 'hit'
        elif player_total == 14:
            return 'stand' if dealer_total <= 6 else 'hit'
        elif player_total == 13:
            return 'stand' if dealer_total <= 6 else 'hit'
        elif player_total == 12:
            # Stand against dealer 4-6, otherwise hit
            return 'stand' if 4 <= dealer_total <= 6 else 'hit'
        
        # Default to hitting in uncertain situations
        return 'hit'
    
    def _get_dealer_turn_action(self, player_total, dealer_total, dealer_bust_prob, has_ace):
        """
        Determine action during dealer's turn: surrender or continue.
        
        Args:
            player_total: Current total of player's cards
            dealer_total: Current total of dealer's visible cards
            dealer_bust_prob: Probability of dealer busting
            has_ace: Whether the player has an ace counted as 11
            
        Returns:
            str: 'surrender' or 'continue'
        """
        # If we have blackjack (21 with 2 cards), always continue
        if player_total == 21 and len(self.player_cards) == 2:
            return 'continue'
            
        # Dealer must stand on 17+, so continue because we know the outcome
        if dealer_total >= 17:
            return 'continue'
        
        # If our total is very low, surrender
        if player_total <= 11:
            return 'surrender'
            
        # If dealer has a high chance of busting, continue
        if dealer_bust_prob > 0.25:
            return 'continue'
        
        # If we have a soft hand (ace), we're in better shape
        if has_ace and player_total >= 18:
            return 'continue'
            
        # Calculate win probability based on current state
        win_prob = self._calculate_win_probability(player_total, dealer_total, has_ace)
        
        # Standard mathematical threshold for surrender:
        # If win probability is < 33.3%, surrendering (keeping half our bet) is better
        surrender_threshold = 0.33
        
        # Table-specific adjustments for surrender threshold
        if self.table_index == 2:  # High dealer bust probability, be more aggressive
            surrender_threshold = 0.30
        elif self.table_index == 4:  # Low dealer bust probability, be more conservative
            surrender_threshold = 0.35
            
        if win_prob < surrender_threshold:
            return 'surrender'
        
        return 'continue'
    
    def _predict_next_card(self, spy_history, is_player=True):
        """
        Predict the next card based on spy history.
        
        Args:
            spy_history: List of spy values
            is_player: Whether prediction is for player (True) or dealer (False)
            
        Returns:
            int: Predicted next card value or None if insufficient history
        """
        if len(spy_history) < 5:
            return None
            
        if is_player:
            next_spy = self.spy_player.get_player_spy_prediction(np.array(spy_history[-5:]))
        else:
            next_spy = self.spy_player.get_dealer_spy_prediction(np.array(spy_history[-5:]))
            
        return get_card_value_from_spy_value(next_spy, is_player=is_player)
    
    def _calculate_win_probability(self, player_total, dealer_total, has_ace):
        """
        Calculate probability of winning given current totals.
        
        Args:
            player_total: Current total of player's cards
            dealer_total: Current total of dealer's visible cards
            has_ace: Whether the player has an ace counted as 11
            
        Returns:
            float: Probability of winning (0.0 to 1.0)
        """
        # Calculate dealer's outcome probabilities
        dealer_outcomes = self._simulate_dealer_outcomes(dealer_total)
        
        # Calculate probability of winning
        win_prob = 0.0
        push_prob = 0.0
        
        for outcome, prob in dealer_outcomes.items():
            # Dealer busts
            if outcome > 21:
                win_prob += prob
            # Player's total exceeds dealer's
            elif outcome < player_total:
                win_prob += prob
            # Push (tie)
            elif outcome == player_total:
                push_prob += prob
        
        # Half of pushes counted as wins for EV calculation
        adjusted_win_prob = win_prob + (push_prob * 0.5)
        
        # Apply table-specific adjustments
        table_factor = {
            0: 1.0,   # Neutral
            1: 0.95,  # Slightly unfavorable
            2: 1.05,  # Slightly favorable (high dealer bust rate)
            3: 1.0,   # Neutral
            4: 0.9    # Unfavorable (low dealer bust rate, high dealer final total)
        }.get(self.table_index, 1.0)
        
        return adjusted_win_prob * table_factor
    
    def _simulate_dealer_outcomes(self, dealer_total):
        """
        Simulate possible dealer outcomes to estimate probability distribution.
        
        Args:
            dealer_total: Current total of dealer's visible cards
            
        Returns:
            dict: Mapping from dealer final total to probability
        """
        outcomes = {}
        
        # Improve dealer outcome simulation using our card prediction
        next_card = self._predict_next_card(self.dealer_spy_values, is_player=False)
        
        # If dealer has 17+, they must stand
        if dealer_total >= 17:
            outcomes[dealer_total] = 1.0
            return outcomes
            
        # If we have a specific next card prediction, use it
        if next_card is not None:
            new_total = dealer_total + next_card
            
            # Dealer takes this card then follows fixed strategy
            if new_total > 21:
                # Dealer busts
                outcomes[new_total] = 1.0
            elif new_total >= 17:
                # Dealer stands
                outcomes[new_total] = 1.0
            else:
                # Dealer still needs more cards - use statistical distribution
                # Calculate weighted probabilities based on dealer_total
                if new_total <= 11:
                    # Very unlikely to bust from this position
                    outcomes[17] = 0.2    # Exactly 17
                    outcomes[18] = 0.25   # Exactly 18
                    outcomes[19] = 0.25   # Exactly 19
                    outcomes[20] = 0.15   # Exactly 20
                    outcomes[21] = 0.1    # Exactly 21
                    outcomes[22] = 0.05   # Bust
                elif new_total <= 13:
                    # Still low chance of busting
                    outcomes[17] = 0.15
                    outcomes[18] = 0.2
                    outcomes[19] = 0.2
                    outcomes[20] = 0.15
                    outcomes[21] = 0.1
                    outcomes[22] = 0.2    # Moderate bust chance
                elif new_total <= 15:
                    # Higher chance of busting
                    outcomes[17] = 0.15
                    outcomes[18] = 0.15
                    outcomes[19] = 0.15
                    outcomes[20] = 0.1
                    outcomes[21] = 0.05
                    outcomes[22] = 0.4    # Significant bust chance
                elif new_total == 16:
                    # Very high chance of busting
                    outcomes[17] = 0.1
                    outcomes[18] = 0.05
                    outcomes[19] = 0.05
                    outcomes[20] = 0.05
                    outcomes[21] = 0.05
                    outcomes[22] = 0.7    # High bust chance
                
                # Table-specific adjustments from dealer_doom analysis
                if self.table_index in [0, 1, 3, 4]:
                    # These tables tend to have lower dealer bust rates
                    for total in list(outcomes.keys()):
                        if total > 21:  # Bust outcomes
                            outcomes[total] *= 0.85  # Reduce bust probability by 15%
                            
                            # Redistribute to standing totals
                            remaining = (1.0 - sum(outcomes.values())) / 5
                            for standing in range(17, 22):
                                if standing in outcomes:
                                    outcomes[standing] += remaining
                                else:
                                    outcomes[standing] = remaining
        else:
            # Use statistical estimates based on current dealer total
            # These are derived from the dealer_bust_probs and expected_totals
            if dealer_total <= 11:
                outcomes[17] = 0.15
                outcomes[18] = 0.2
                outcomes[19] = 0.25
                outcomes[20] = 0.2
                outcomes[21] = 0.1
                outcomes[22] = 0.1
            elif dealer_total == 12:
                outcomes[17] = 0.15
                outcomes[18] = 0.15
                outcomes[19] = 0.15
                outcomes[20] = 0.15
                outcomes[21] = 0.1
                outcomes[22] = 0.3
            elif dealer_total == 13:
                outcomes[17] = 0.15
                outcomes[18] = 0.15
                outcomes[19] = 0.15
                outcomes[20] = 0.1
                outcomes[21] = 0.05
                outcomes[22] = 0.4
            elif dealer_total == 14:
                outcomes[17] = 0.15
                outcomes[18] = 0.15
                outcomes[19] = 0.1
                outcomes[20] = 0.05
                outcomes[21] = 0.05
                outcomes[22] = 0.5
            elif dealer_total == 15:
                outcomes[17] = 0.1
                outcomes[18] = 0.1
                outcomes[19] = 0.1
                outcomes[20] = 0.05
                outcomes[21] = 0.05
                outcomes[22] = 0.6
            elif dealer_total == 16:
                outcomes[17] = 0.1
                outcomes[18] = 0.05
                outcomes[19] = 0.05
                outcomes[20] = 0.05
                outcomes[21] = 0.05
                outcomes[22] = 0.7
            
            # Adjust by the table-specific bust probability
            if self.table_index in self.dealer_bust_probs:
                bust_prob = self.dealer_bust_probs[self.table_index]
                expected_bust = outcomes.get(22, 0.0)
                
                # Scale the bust probability
                if expected_bust > 0:
                    scale_factor = bust_prob / expected_bust
                    outcomes[22] = bust_prob
                    
                    # Ensure total probability is 1.0
                    remaining = 1.0 - bust_prob
                    total_non_bust = sum([p for k, p in outcomes.items() if k != 22])
                    
                    if total_non_bust > 0:
                        for k in outcomes.keys():
                            if k != 22:
                                outcomes[k] = outcomes[k] * (remaining / total_non_bust)
        
        # Normalize probabilities to ensure they sum to 1.0
        total_prob = sum(outcomes.values())
        if total_prob > 0:
            for k in outcomes:
                outcomes[k] /= total_prob
                
        return outcomes

if __name__ == "__main__":
    print("One Time Showdown - Solution Test")
    print("=" * 50)
    
    # Test with a sample game
    player = MyPlayer(0)
    
    # Sample game state 1: Player with 16, dealer showing 10
    action = player.get_player_action(
        [10, 6], [10.0, 6.0],
        [10], [10.0],
        16, 10, 'player'
    )
    print(f"Player's turn with 16 vs dealer 10: {action}")
    
    # Sample game state 2: Dealer turn decision
    action = player.get_player_action(
        [10, 5], [10.0, 5.0],
        [10], [10.0],
        15, 10, 'dealer'
    )
    print(f"Dealer's turn with player 15 vs dealer 10: {action}")
    
    # Test different tables
    for table in range(5):
        player = MyPlayer(table)
        action = player.get_player_action(
            [10, 4], [10.0, 4.0],
            [6], [6.0],
            14, 6, 'player'
        )
        print(f"Table {table} - Player's turn with 14 vs dealer 6: {action}") 