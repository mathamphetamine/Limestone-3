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
            
            # First check if train_split exists for better testing
            train_split_path = os.path.join(repo_root, 'test_data', 'train_split.csv')
            if os.path.exists(train_split_path):
                data_path = train_split_path
                print(f"Using train split for model training: {data_path}")
            else:
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
            # Table 2: Complex data with high variance
            # Improved model for high MSE in Table 2
            self.player_model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, 
                                                        max_depth=4, random_state=42,
                                                        subsample=0.8, min_samples_split=5)
            
            # For Table 2, analyze the extreme value patterns
            if len(self.player_spy) > 10:
                # Calculate statistics on extreme values
                extreme_high_values = self.player_spy[self.player_spy > 200]
                extreme_low_values = self.player_spy[self.player_spy < -200]
                
                if len(extreme_high_values) > 5:
                    # Find the average change after extreme high values
                    extreme_high_indices = np.where(self.player_spy > 200)[0]
                    high_next_indices = extreme_high_indices + 1
                    high_next_indices = high_next_indices[high_next_indices < len(self.player_spy)]
                    
                    if len(high_next_indices) > 0:
                        high_values = self.player_spy[extreme_high_indices[high_next_indices - extreme_high_indices == 1]]
                        next_values = self.player_spy[high_next_indices]
                        self.high_value_ratio = np.mean(next_values / high_values) if len(next_values) > 0 else 0.3
                    else:
                        self.high_value_ratio = 0.3
                else:
                    self.high_value_ratio = 0.3
                    
                if len(extreme_low_values) > 5:
                    # Find the average change after extreme low values
                    extreme_low_indices = np.where(self.player_spy < -200)[0]
                    low_next_indices = extreme_low_indices + 1
                    low_next_indices = low_next_indices[low_next_indices < len(self.player_spy)]
                    
                    if len(low_next_indices) > 0:
                        low_values = self.player_spy[extreme_low_indices[low_next_indices - extreme_low_indices == 1]]
                        next_values = self.player_spy[low_next_indices]
                        self.low_value_ratio = np.mean(next_values / low_values) if len(next_values) > 0 else 0.4
                    else:
                        self.low_value_ratio = 0.4
                else:
                    self.low_value_ratio = 0.4
            else:
                self.high_value_ratio = 0.3
                self.low_value_ratio = 0.4
                
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
            hist (list): List of past spy values
            
        Returns:
            float: Predicted next spy value
        """
        if len(hist) < 5:
            # If not enough history, use mean of available values
            return np.mean(hist) if hist else 0
        
        # Special case for alternating patterns
        if self.player_alternating and len(hist) >= 2:
            last_diff = hist[-1] - hist[-2]
            return hist[-1] - last_diff
        
        # Convert history to features
        X = np.array(hist[-5:]).reshape(1, -1)
        
        # Table-specific predictions
        if self.table_index == 0:
            prediction = self.player_model.predict(X)[0]
        elif self.table_index == 1:
            if len(hist) >= 2:
                # Try to use card transition data if available
                last_card = None
                curr_card = None
                
                # Convert recent spy values to likely cards
                for i in range(len(hist) - 1, max(len(hist) - 3, -1), -1):
                    rounded_spy = round(hist[i])
                    if rounded_spy in self.player_spy_to_card:
                        if curr_card is None:
                            curr_card = self.player_spy_to_card[rounded_spy]
                        elif last_card is None:
                            last_card = self.player_spy_to_card[rounded_spy]
                            break
                
                # If we found a matching transition, use its mean
                if last_card is not None and curr_card is not None:
                    key = f"{last_card}->{curr_card}"
                    if key in self.transition_means:
                        return self.transition_means[key]
            
            # Fall back to model prediction
            prediction = self.player_model.predict(X)[0]
        elif self.table_index == 2:
            # Improved prediction for Table 2 using a weighted ensemble approach
            # First get the model prediction
            model_prediction = self.player_model.predict(X)[0]
            
            # Calculate exponential moving average
            alpha = 0.3
            ema = 0
            for i in range(len(hist)):
                ema = alpha * hist[i] + (1 - alpha) * ema
            
            # Get simple moving average of last 3 values
            sma = np.mean(hist[-3:]) if len(hist) >= 3 else np.mean(hist)
            
            # Check for high volatility patterns
            volatility = np.std(hist[-5:]) if len(hist) >= 5 else 1.0
            
            # Special case handling for very large values and extreme volatility
            last_value = hist[-1]
            if last_value > 200:
                # Extreme high values tend to drop significantly
                return last_value * self.high_value_ratio
            elif last_value < -200:
                # Extreme negative values tend to rebound
                return last_value * self.low_value_ratio
            elif volatility > 150:
                # Very high volatility: use EMA with stronger weighting
                prediction = 0.5 * model_prediction + 0.5 * ema
            elif volatility > 100:
                # High volatility: use model with EMA stabilization
                prediction = 0.7 * model_prediction + 0.3 * ema
            else:
                # Lower volatility: blend model with moving average
                prediction = 0.8 * model_prediction + 0.2 * sma
        elif self.table_index == 3:
            # Table 3 uses card transition effects
            prediction = self.player_model.predict(X)[0]
            
            # Apply card transition effects if we can identify them
            if len(hist) >= 2:
                last_rounded = round(hist[-1] / 10) * 10
                prev_rounded = round(hist[-2] / 10) * 10
                
                if last_rounded in self.player_spy_to_card and prev_rounded in self.player_spy_to_card:
                    last_card = self.player_spy_to_card[last_rounded]
                    prev_card = self.player_spy_to_card[prev_rounded]
                    
                    key = f"{prev_card}->{last_card}"
                    if key in self.card_transitions and len(self.card_transitions[key]) >= 5:
                        transition_effect = np.mean(self.card_transitions[key])
                        prediction = hist[-1] + transition_effect
        else:  # Table 4
            prediction = self.player_model.predict(X)[0]
        
        return prediction
    
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