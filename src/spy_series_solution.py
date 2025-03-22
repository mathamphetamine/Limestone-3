import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

class MyPlayer:
    """
    The MyPlayer class implements the solution for the Spy Series Seer challenge.
    
    It provides methods to predict the next spy value in a time series for both 
    player and dealer based on historical data. The implementation includes
    table-specific optimizations, pattern detection, and machine learning models.
    
    Each of the 5 tables has unique characteristics that require different
    prediction approaches, which are selected and configured based on
    statistical analysis.
    """
    
    def __init__(self, table_index):
        """
        Initialize the MyPlayer instance with table-specific data and models.
        
        Args:
            table_index (int): The index of the table (0-4) to analyze and predict.
        """
        self.table_index = table_index
        self.player_model = None
        self.dealer_model = None
        self.player_spy = []
        self.dealer_spy = []
        self.player_card = []
        self.dealer_card = []
        self.player_alternating = False
        self.dealer_alternating = False
        
        # Additional data structures for Table 3 card sequence prediction
        if table_index == 3:
            self.card_transitions = {}
        
        # Load and prepare data
        try:
            # Find the data directory relative to current file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            repo_root = os.path.dirname(current_dir)
            data_path = os.path.join(repo_root, 'data', 'train.csv')
            
            data = pd.read_csv(data_path, header=[0,1,2])
            player_data = np.array([
                data[(f'table_{table_index}', 'player', 'spy')],
                data[(f'table_{table_index}', 'player', 'card')]
            ]).T
            
            dealer_data = np.array([
                data[(f'table_{table_index}', 'dealer', 'spy')],
                data[(f'table_{table_index}', 'dealer', 'card')]
            ]).T
            
            self.player_spy = player_data[:, 0]
            self.player_card = player_data[:, 1]
            self.dealer_spy = dealer_data[:, 0]
            self.dealer_card = dealer_data[:, 1]
            
            # Analyze sequential patterns
            self._analyze_patterns()
            
            # Create card mappings
            self._create_spy_card_mappings()
            
            # Train prediction models
            self._train_models()
        except Exception as e:
            print(f"Error initializing MyPlayer: {e}")
    
    def _analyze_patterns(self):
        """
        Analyze patterns in the sequence data for both player and dealer.
        
        This method:
        1. Detects alternating patterns in spy values
        2. Calculates autocorrelations
        3. Analyzes card transitions for Table 1
        """
        
        # Check for alternating patterns in dealer spy values
        if len(self.dealer_spy) > 10:
            diffs = np.diff(self.dealer_spy)
            alternating = np.all(diffs[0::2] > 0) and np.all(diffs[1::2] < 0)
            alternating = alternating or (np.all(diffs[0::2] < 0) and np.all(diffs[1::2] > 0))
            self.dealer_alternating = alternating
            
        # Check for alternating patterns in player spy values
        if len(self.player_spy) > 10:
            diffs = np.diff(self.player_spy)
            alternating = np.all(diffs[0::2] > 0) and np.all(diffs[1::2] < 0)
            alternating = alternating or (np.all(diffs[0::2] < 0) and np.all(diffs[1::2] > 0))
            self.player_alternating = alternating
            
        # Calculate autocorrelations
        self.player_autocorrelations = []
        if len(self.player_spy) > 5:
            for lag in range(1, 6):
                if len(self.player_spy) > lag:
                    corr = np.corrcoef(self.player_spy[:-lag], self.player_spy[lag:])[0, 1]
                    self.player_autocorrelations.append(corr)
            
        # For Table 1, create card transition mapping
        if self.table_index == 1:
            self.card_transitions = {}
            for i in range(1, len(self.player_card)):
                prev_card = self.player_card[i-1]
                curr_card = self.player_card[i]
                key = f"{prev_card}->{curr_card}"
                
                if key not in self.card_transitions:
                    self.card_transitions[key] = []
                
                # Store the spy value for this transition
                self.card_transitions[key].append(self.player_spy[i])
            
            # Calculate mean spy value for each transition
            self.transition_means = {}
            self.transition_stds = {}
            for key, values in self.card_transitions.items():
                if len(values) >= 10:  # Only consider transitions with enough data
                    self.transition_means[key] = np.mean(values)
                    self.transition_stds[key] = np.std(values)
    
    def _create_spy_card_mappings(self):
        """
        Create mappings between spy values and cards.
        
        This method:
        1. Maps player cards to average spy values
        2. Maps player spy values to most common corresponding cards
        3. Does the same for dealer cards/spy values
        4. Creates table-specific mappings with appropriate rounding strategies
        """
        # Player card to spy mapping
        self.player_card_to_spy = {}
        for card in sorted(set(self.player_card)):
            mask = (self.player_card == card)
            self.player_card_to_spy[card] = np.mean(self.player_spy[mask])
            
        # Player spy to card mapping (for precise prediction)
        self.player_spy_to_card = {}
        
        # Different rounding approach based on table
        if self.table_index == 3:
            player_rounded_spy = np.round(self.player_spy / 10) * 10
        elif self.table_index == 1:
            player_rounded_spy = np.round(self.player_spy)
        else:
            player_rounded_spy = np.round(self.player_spy)
        
        for rounded_spy in np.unique(player_rounded_spy):
            mask = (player_rounded_spy == rounded_spy)
            cards = self.player_card[mask]
            unique_cards, counts = np.unique(cards, return_counts=True)
            self.player_spy_to_card[rounded_spy] = unique_cards[np.argmax(counts)]
            
        # Dealer card to spy mapping
        self.dealer_card_to_spy = {}
        for card in sorted(set(self.dealer_card)):
            mask = (self.dealer_card == card)
            self.dealer_card_to_spy[card] = np.mean(self.dealer_spy[mask])
        
        # Dealer spy to card mapping
        self.dealer_spy_to_card = {}
        
        # Different rounding approach based on table
        if self.table_index == 1:
            dealer_rounded_spy = np.round(self.dealer_spy)
        else:
            dealer_rounded_spy = np.round(self.dealer_spy)
        
        for rounded_spy in np.unique(dealer_rounded_spy):
            mask = (dealer_rounded_spy == rounded_spy)
            cards = self.dealer_card[mask]
            unique_cards, counts = np.unique(cards, return_counts=True)
            self.dealer_spy_to_card[rounded_spy] = unique_cards[np.argmax(counts)]
        
        # For Table 3, build card transition effects
        if self.table_index == 3:
            self.card_transitions = {}
            for i in range(1, len(self.player_card)):
                prev_card = self.player_card[i-1]
                curr_card = self.player_card[i]
                key = f"{prev_card}->{curr_card}"
                
                if key not in self.card_transitions:
                    self.card_transitions[key] = []
                
                # Store the spy value change
                self.card_transitions[key].append(self.player_spy[i] - self.player_spy[i-1])
    
    def _train_models(self):
        """
        Train predictive models for spy values.
        
        This method:
        1. Creates time series data for training
        2. Selects appropriate models based on table characteristics
        3. Trains models on the prepared data
        
        Each table uses different model types based on MSE analysis.
        """
        X_player, y_player = self._create_time_series_data(self.player_spy)
        X_dealer, y_dealer = self._create_time_series_data(self.dealer_spy)
        
        # Select models based on table characteristics and our MSE analysis
        if self.table_index == 0:
            # Table 0: Very consistent patterns
            self.player_model = LinearRegression()
            self.dealer_model = "alternating"  # Special case for alternating pattern
        elif self.table_index == 1:
            # Table 1: Moderate autocorrelation with polynomial patterns
            self.player_model = make_pipeline(PolynomialFeatures(degree=2), Ridge(alpha=0.01))
            
            # For table 1, integer dealer spy values map directly to cards with high accuracy
            self.dealer_model = GradientBoostingRegressor(n_estimators=100, max_depth=3)
        elif self.table_index == 2:
            # Table 2: Complex data with range-based behaviors
            self.player_model = RandomForestRegressor(n_estimators=100, random_state=42) 
            self.dealer_model = make_pipeline(PolynomialFeatures(degree=2), Ridge(alpha=0.1))
        elif self.table_index == 3:
            # Table 3: High autocorrelation data
            self.player_model = GradientBoostingRegressor(n_estimators=100, max_depth=3)
            self.dealer_model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
        elif self.table_index == 4:
            # Table 4: Moderate variance with strong polynomial relationships
            self.player_model = make_pipeline(PolynomialFeatures(degree=3), Ridge(alpha=0.1))
            self.dealer_model = "alternating"
        
        # Fit models if they aren't using the alternating pattern
        if self.player_model != "alternating":
            self.player_model.fit(X_player, y_player)
        
        if self.dealer_model != "alternating":
            self.dealer_model.fit(X_dealer, y_dealer)
    
    def _create_time_series_data(self, series, window_size=5):
        """
        Create windowed time series data for prediction.
        
        Args:
            series (numpy.ndarray): The time series data to window
            window_size (int): The size of the window to use (default: 5)
            
        Returns:
            tuple: X (features) and y (targets) for time series prediction
        """
        X, y = [], []
        for i in range(len(series) - window_size):
            X.append(series[i:i+window_size])
            y.append(series[i+window_size])
        return np.array(X), np.array(y)
    
    def get_card_value_from_spy_value(self, value, is_player=True):
        """
        Get the corresponding card value for a spy value.
        
        Args:
            value (float): The spy value to convert to a card value
            is_player (bool): Whether this is for player (True) or dealer (False)
            
        Returns:
            int: The predicted card value
        """
        if is_player:
            # Round to appropriate precision for this table
            if self.table_index == 3:
                rounded_value = round(value / 10) * 10
            elif self.table_index == 1:
                rounded_value = round(value)
            else:
                rounded_value = round(value)
            
            # Check if we have this spy value in our mapping
            if rounded_value in self.player_spy_to_card:
                return self.player_spy_to_card[rounded_value]
            
            # Fallback to closest card
            return min(self.player_spy_to_card.keys(), 
                      key=lambda x: abs(x - rounded_value))
        else:
            # Round to appropriate precision for this table
            if self.table_index == 1:
                rounded_value = round(value)
            else:
                rounded_value = round(value)
            
            # Check if we have this spy value in our mapping
            if rounded_value in self.dealer_spy_to_card:
                return self.dealer_spy_to_card[rounded_value]
            
            # Fallback to closest card
            return min(self.dealer_spy_to_card.keys(), 
                      key=lambda x: abs(x - rounded_value))
    
    def get_player_spy_prediction(self, hist):
        """
        Predict the next player spy value based on historical data.
        
        Args:
            hist (numpy.ndarray): Array of historical spy values (length: 5)
            
        Returns:
            float: Predicted next spy value for the player
        """
        # Table-specific optimizations
        if self.table_index == 1:
            # Table 1: Use the fitted model but with some adjustments
            model_pred = self.player_model.predict(hist.reshape(1, -1))[0]
            
            # Bound predictions to reasonable range based on analysis
            # Table 1 player spy values fall between 70.55 and 89.03
            return np.clip(model_pred, 70.5, 89.0)
            
        elif self.table_index == 2:
            # Range-based prediction for Table 2
            last_value = hist[-1]
            
            # Models based on the current value range
            if 50 <= last_value < 100:
                # This range has the lowest MSE (326.9556)
                # Average change in this range is -17.1879
                return last_value - 17.19
            elif last_value >= 100:
                # Values in this range tend to decrease dramatically
                # Average change is -126.423
                return last_value - 126.42
            elif 0 <= last_value < 50:
                # Use median of last values for this range
                if len(hist) >= 10:
                    return np.median(hist[-10:])
                else:
                    # When not enough history, predict slight increase
                    return last_value + 10.74
            else:
                # Fallback
                return last_value
        elif self.table_index == 3:
            # Table 3 has extremely high autocorrelation (0.9997)
            # Use a very simple model that takes advantage of this property
            if len(hist) < 3:
                return hist[-1]  # Not enough history
                
            # Table 3 analysis showed that the best model was "Mean of Last 3" with MSE = 48.26
            return np.mean(hist[-3:])
            
        # Check for alternating pattern
        if (self.table_index == 4 or self.player_alternating) and len(hist) > 1:
            diffs = np.diff(hist)
            last_diff = diffs[-1]
            # If pattern shows strong alternating behavior
            if np.std(diffs) < 1.0 and abs(last_diff) > 0.1:
                return hist[-1] - last_diff
        
        if self.player_model == "alternating":
            # Handle alternating pattern
            diffs = np.diff(hist)
            if len(diffs) > 0:
                return hist[-1] - diffs[-1]
            else:
                return hist[-1]
        else:
            return self.player_model.predict(hist.reshape(1, -1))[0]
    
    def get_dealer_spy_prediction(self, hist):
        """
        Predict the next dealer spy value based on historical data.
        
        Args:
            hist (numpy.ndarray): Array of historical spy values (length: 5)
            
        Returns:
            float: Predicted next spy value for the dealer
        """
        # Table-specific optimizations
        if self.table_index == 1:
            # For table 1, integer dealer spy values map to specific cards
            # Round the prediction to the nearest integer
            if self.dealer_model != "alternating":
                pred = self.dealer_model.predict(hist.reshape(1, -1))[0]
                return round(pred)
        
        if self.dealer_model == "alternating" or self.dealer_alternating:
            # Handle alternating pattern
            diffs = np.diff(hist)
            if len(diffs) > 0:
                # For strong alternating patterns, negate the last difference
                if self.table_index == 0 or self.table_index == 4:
                    return hist[-1] - diffs[-1]
                else:
                    return hist[-1] + (-1 * np.mean(diffs))
            else:
                # Fallback if not enough history
                return hist[-1]
        else:
            return self.dealer_model.predict(hist.reshape(1, -1))[0] 