"""
Transaction Cost Models for FX Trading
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional


class FXCosts:
    """Foreign Exchange transaction cost model"""
    
    def __init__(self):
        # Typical spreads in pips for different pair types
        self.spread_table = {
            # Major pairs (tight spreads)
            'EURUSD': 0.5,
            'USDJPY': 0.5,
            'GBPUSD': 0.7,
            'AUDUSD': 0.6,
            'USDCAD': 0.8,
            'NZDUSD': 0.9,
            'USDCHF': 0.8,
            
            # Cross pairs (wider spreads)
            'EURJPY': 1.0,
            'GBPJPY': 1.2,
            'AUDJPY': 1.0,
            'EURGBP': 1.0,
            'EURAUD': 1.2,
            'AUDNZD': 1.5,
            'GBPAUD': 1.8,
            'NZDJPY': 1.5,
            
            # Minor/Exotic (even wider)
            'USDTRY': 15.0,
            'USDZAR': 20.0,
            'USDMXN': 10.0,
            
            # Default for unknown pairs
            'default': 1.0
        }
        
        # Pip sizes
        self.pip_sizes = {
            'JPY': 0.01,     # For XXX/JPY pairs
            'default': 0.0001 # For all others
        }
        
    def get_spread_pips(self, pair: str) -> float:
        """Get typical spread for a currency pair in pips"""
        return self.spread_table.get(pair, self.spread_table['default'])
    
    def get_pip_size(self, pair: str) -> float:
        """Get pip size for a currency pair"""
        if pair.endswith('JPY'):
            return self.pip_sizes['JPY']
        return self.pip_sizes['default']
    
    def fx_round_turn(self, 
                     pair: str,
                     spread_pips: Optional[float] = None,
                     slippage_bps: float = 0.0,
                     commission_bps: float = 0.0) -> float:
        """
        Calculate round-turn transaction costs in basis points
        
        Parameters:
        -----------
        pair : str
            Currency pair (e.g., 'EURUSD')
        spread_pips : float, optional
            Spread in pips (if None, uses typical spread)
        slippage_bps : float
            Additional slippage in basis points
        commission_bps : float
            Commission in basis points (if any)
            
        Returns:
        --------
        float : Total round-turn cost in basis points
        """
        
        # Get spread if not provided
        if spread_pips is None:
            spread_pips = self.get_spread_pips(pair)
            
        # Convert spread from pips to basis points
        # 1 pip = 0.0001 = 1 basis point for most pairs
        # 1 pip = 0.01 = 100 basis points for JPY pairs
        pip_size = self.get_pip_size(pair)
        spread_bps = (spread_pips * pip_size) * 10000
        
        # Total round-turn cost
        total_cost_bps = spread_bps + slippage_bps + commission_bps
        
        return total_cost_bps
    
    def apply_costs_to_trades(self,
                            trades: pd.DataFrame,
                            pair: str,
                            entry_slippage_bps: float = 1.0,
                            exit_slippage_bps: float = 1.0) -> pd.DataFrame:
        """
        Apply transaction costs to a DataFrame of trades
        
        Parameters:
        -----------
        trades : pd.DataFrame
            Must have columns: entry_price, exit_price, position (1 or -1)
        pair : str
            Currency pair
        entry_slippage_bps : float
            Slippage on entry
        exit_slippage_bps : float
            Slippage on exit
            
        Returns:
        --------
        pd.DataFrame with adjusted prices and costs
        """
        
        trades = trades.copy()
        
        # Get costs
        spread_pips = self.get_spread_pips(pair)
        pip_size = self.get_pip_size(pair)
        half_spread = (spread_pips * pip_size) / 2
        
        # Adjust entry prices
        trades['entry_price_adjusted'] = trades.apply(
            lambda row: row['entry_price'] + half_spread if row['position'] > 0 
                       else row['entry_price'] - half_spread,
            axis=1
        )
        
        # Add entry slippage
        entry_slip = entry_slippage_bps / 10000
        trades['entry_price_adjusted'] *= (1 + entry_slip * trades['position'])
        
        # Adjust exit prices
        trades['exit_price_adjusted'] = trades.apply(
            lambda row: row['exit_price'] - half_spread if row['position'] > 0
                       else row['exit_price'] + half_spread,
            axis=1
        )
        
        # Add exit slippage
        exit_slip = exit_slippage_bps / 10000
        trades['exit_price_adjusted'] *= (1 - exit_slip * trades['position'])
        
        # Calculate cost impact
        trades['entry_cost_bps'] = abs(
            (trades['entry_price_adjusted'] - trades['entry_price']) / 
            trades['entry_price']
        ) * 10000
        
        trades['exit_cost_bps'] = abs(
            (trades['exit_price_adjusted'] - trades['exit_price']) / 
            trades['exit_price']
        ) * 10000
        
        trades['total_cost_bps'] = trades['entry_cost_bps'] + trades['exit_cost_bps']
        
        # Adjusted returns
        trades['gross_return'] = (
            trades['exit_price'] - trades['entry_price']
        ) / trades['entry_price'] * trades['position']
        
        trades['net_return'] = (
            trades['exit_price_adjusted'] - trades['entry_price_adjusted']
        ) / trades['entry_price_adjusted'] * trades['position']
        
        trades['cost_impact'] = trades['gross_return'] - trades['net_return']
        
        return trades
    
    def estimate_market_impact(self,
                             trade_size_usd: float,
                             adv_usd: float,
                             urgency: str = 'medium') -> float:
        """
        Estimate market impact based on trade size vs average daily volume
        
        Parameters:
        -----------
        trade_size_usd : float
            Trade size in USD
        adv_usd : float
            Average daily volume in USD
        urgency : str
            'low', 'medium', 'high'
            
        Returns:
        --------
        float : Estimated market impact in basis points
        """
        
        # Participation rate
        participation = trade_size_usd / adv_usd
        
        # Base impact (square root model)
        base_impact = 10 * np.sqrt(participation)  # 10 bps for 1% of ADV
        
        # Urgency multiplier
        urgency_mult = {
            'low': 0.5,    # Passive execution
            'medium': 1.0,  # Normal
            'high': 2.0     # Aggressive
        }
        
        impact_bps = base_impact * urgency_mult.get(urgency, 1.0)
        
        return impact_bps


def calculate_slippage_curve(pair: str,
                           trade_sizes: list,
                           adv_usd: float = 1e9) -> pd.DataFrame:
    """
    Calculate slippage for different trade sizes
    
    Parameters:
    -----------
    pair : str
        Currency pair
    trade_sizes : list
        List of trade sizes as multiples of ADV
    adv_usd : float
        Average daily volume in USD
        
    Returns:
    --------
    pd.DataFrame with slippage estimates
    """
    
    costs = FXCosts()
    
    results = []
    for size_mult in trade_sizes:
        trade_size = size_mult * adv_usd
        
        # Fixed costs
        spread_cost = costs.fx_round_turn(pair)
        
        # Variable costs (market impact)
        impact = costs.estimate_market_impact(trade_size, adv_usd)
        
        # Total slippage
        total_slippage = spread_cost + impact
        
        results.append({
            'size_multiple': size_mult,
            'trade_size_usd': trade_size,
            'spread_cost_bps': spread_cost,
            'market_impact_bps': impact,
            'total_slippage_bps': total_slippage,
            'total_slippage_pct': total_slippage / 100
        })
    
    return pd.DataFrame(results)