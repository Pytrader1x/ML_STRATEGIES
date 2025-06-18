"""
Capacity and Market Impact Analysis
Determines maximum tradeable size while maintaining target Sharpe ratio
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from pathlib import Path
import json
import logging
from dataclasses import dataclass

import sys
sys.path.append('..')

from quantlab import momentum, Backtest
from quantlab.costs import FXCosts, calculate_slippage_curve

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MarketMicrostructure:
    """FX market microstructure parameters"""
    pair: str
    avg_daily_volume_usd: float
    typical_spread_pips: float
    tick_size: float
    session_volumes: Dict[str, float]  # % of volume by session
    

class CapacityAnalyzer:
    """Analyze strategy capacity and market impact"""
    
    # Typical ADV for major FX pairs (in billions USD)
    FX_ADV = {
        'EURUSD': 1500,   # Most liquid
        'USDJPY': 900,
        'GBPUSD': 650,
        'AUDUSD': 350,
        'USDCAD': 280,
        'NZDUSD': 120,
        'USDCHF': 180,
        'EURJPY': 100,
        'GBPJPY': 80,
        'AUDJPY': 60,
        'EURGBP': 90,
        'AUDNZD': 20,
        'default': 50
    }
    
    def __init__(self, data: pd.DataFrame, pair: str):
        self.data = data.copy()
        self.pair = pair
        self.costs = FXCosts()
        self.base_params = {'lookback': 40, 'entry_z': 1.5, 'exit_z': 0.5}
        
        # Get market parameters
        self.adv_usd = self.FX_ADV.get(pair, self.FX_ADV['default']) * 1e9
        self.pip_size = self.costs.get_pip_size(pair)
        self.spread_pips = self.costs.get_spread_pips(pair)
        
    def analyze_capacity(self,
                        size_multiples: List[float] = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1],
                        target_sharpe: float = 1.0) -> pd.DataFrame:
        """
        Analyze strategy capacity at different size levels
        
        Parameters:
        -----------
        size_multiples : list
            Trade sizes as fraction of ADV
        target_sharpe : float
            Minimum acceptable Sharpe ratio
            
        Returns:
        --------
        pd.DataFrame with capacity analysis
        """
        
        logger.info(f"Analyzing capacity for {self.pair}")
        logger.info(f"ADV: ${self.adv_usd/1e9:.1f}B")
        
        results = []
        
        # Generate base signals
        signals = momentum(self.data['Close'], **self.base_params)
        
        # Get base performance (no impact)
        backtest = Backtest(self.data)
        base_result = backtest.run(signals['signal'], self.pair)
        base_sharpe = base_result.metrics['sharpe_ratio']
        base_return = base_result.metrics['total_return']
        
        logger.info(f"Base Sharpe (no impact): {base_sharpe:.3f}")
        
        for size_mult in size_multiples:
            trade_size_usd = size_mult * self.adv_usd
            
            # Calculate market impact
            impact_bps = self.costs.estimate_market_impact(
                trade_size_usd, 
                self.adv_usd,
                urgency='medium'
            )
            
            # Total cost including spread and impact
            total_cost_bps = self.costs.fx_round_turn(self.pair) + impact_bps
            
            # Estimate degraded performance
            # Simple model: Sharpe degrades proportionally to costs
            cost_drag = total_cost_bps / 100  # Convert to percentage
            trades_per_year = base_result.metrics['num_trades'] * (252 * 96 / len(self.data))
            annual_cost_drag = cost_drag * trades_per_year
            
            # Adjusted metrics
            adjusted_return = base_return - annual_cost_drag
            adjusted_sharpe = base_sharpe * (1 - annual_cost_drag / base_return) if base_return > 0 else 0
            
            # Calculate break-even size where Sharpe = target
            if base_sharpe > target_sharpe:
                sharpe_buffer = base_sharpe - target_sharpe
                max_cost_drag = (sharpe_buffer / base_sharpe) * base_return
                max_annual_cost = max_cost_drag / trades_per_year
                max_impact_bps = max_annual_cost * 100 - self.costs.fx_round_turn(self.pair)
                
                # Reverse engineer max size
                # impact = 10 * sqrt(size/adv), so size = (impact/10)^2 * adv
                max_size_mult = (max_impact_bps / 10) ** 2 if max_impact_bps > 0 else 0
            else:
                max_size_mult = 0
            
            results.append({
                'size_multiple': size_mult,
                'trade_size_usd': trade_size_usd,
                'trade_size_mm': trade_size_usd / 1e6,
                'spread_cost_bps': self.costs.fx_round_turn(self.pair),
                'market_impact_bps': impact_bps,
                'total_cost_bps': total_cost_bps,
                'base_sharpe': base_sharpe,
                'adjusted_sharpe': adjusted_sharpe,
                'base_return': base_return,
                'adjusted_return': adjusted_return,
                'annual_cost_drag': annual_cost_drag,
                'viable': adjusted_sharpe >= target_sharpe,
                'max_size_multiple': max_size_mult
            })
        
        return pd.DataFrame(results)
    
    def analyze_session_liquidity(self) -> pd.DataFrame:
        """
        Analyze liquidity by trading session
        
        Returns:
        --------
        pd.DataFrame with session analysis
        """
        
        # Define FX sessions (in UTC)
        sessions = {
            'Sydney': (21, 6),    # 21:00 - 06:00 UTC
            'Tokyo': (0, 9),      # 00:00 - 09:00 UTC  
            'London': (7, 16),    # 07:00 - 16:00 UTC
            'New York': (12, 21)  # 12:00 - 21:00 UTC
        }
        
        # Typical volume distribution
        session_volumes = {
            'Sydney': 0.05,
            'Tokyo': 0.15,
            'London': 0.35,
            'New York': 0.30,
            'London/NY Overlap': 0.15
        }
        
        results = []
        
        for session_name, (start_hour, end_hour) in sessions.items():
            # Filter data for session
            if start_hour < end_hour:
                session_mask = (self.data.index.hour >= start_hour) & \
                              (self.data.index.hour < end_hour)
            else:  # Crosses midnight
                session_mask = (self.data.index.hour >= start_hour) | \
                              (self.data.index.hour < end_hour)
            
            session_data = self.data[session_mask]
            
            if len(session_data) < 1000:
                continue
                
            # Generate signals and run backtest
            try:
                signals = momentum(session_data['Close'], **self.base_params)
                backtest = Backtest(session_data)
                result = backtest.run(signals['signal'], self.pair)
                
                # Estimate session liquidity
                if session_name == 'London':
                    session_adv = self.adv_usd * 0.35
                elif session_name == 'New York':
                    session_adv = self.adv_usd * 0.30
                elif session_name == 'Tokyo':
                    session_adv = self.adv_usd * 0.15
                else:
                    session_adv = self.adv_usd * 0.05
                
                results.append({
                    'session': session_name,
                    'hours_utc': f"{start_hour:02d}:00-{end_hour:02d}:00",
                    'sharpe': result.metrics['sharpe_ratio'],
                    'returns': result.metrics['total_return'],
                    'num_trades': result.metrics['num_trades'],
                    'win_rate': result.metrics['win_rate'],
                    'session_adv_usd': session_adv,
                    'session_adv_pct': (session_adv / self.adv_usd) * 100
                })
                
            except Exception as e:
                logger.warning(f"Error analyzing {session_name}: {e}")
                continue
                
        return pd.DataFrame(results)
    
    def analyze_trade_clustering(self) -> Dict:
        """
        Analyze trade clustering and its impact on capacity
        
        Returns:
        --------
        Dict with clustering analysis
        """
        
        # Generate signals
        signals_df = momentum(self.data['Close'], **self.base_params)
        signals = signals_df['signal']
        
        # Find entry points
        entries = signals.diff() != 0
        entry_times = self.data.index[entries & (signals != 0)]
        
        if len(entry_times) < 2:
            return {'error': 'Insufficient trades'}
        
        # Calculate time between trades
        time_diffs = pd.Series(entry_times[1:]) - pd.Series(entry_times[:-1])
        time_diffs_minutes = time_diffs.total_seconds() / 60
        
        # Analyze clustering
        clustering_stats = {
            'total_entries': len(entry_times),
            'avg_time_between_trades_min': time_diffs_minutes.mean(),
            'min_time_between_trades_min': time_diffs_minutes.min(),
            'trades_within_1h': (time_diffs_minutes < 60).sum(),
            'trades_within_15min': (time_diffs_minutes < 15).sum(),
            'max_trades_per_day': 0,
            'clustering_risk': 'Low'
        }
        
        # Count max trades per day
        trades_per_day = pd.Series(entry_times).dt.date.value_counts()
        clustering_stats['max_trades_per_day'] = trades_per_day.max()
        clustering_stats['avg_trades_per_day'] = trades_per_day.mean()
        
        # Assess clustering risk
        if clustering_stats['trades_within_15min'] > len(entry_times) * 0.1:
            clustering_stats['clustering_risk'] = 'High'
        elif clustering_stats['trades_within_1h'] > len(entry_times) * 0.3:
            clustering_stats['clustering_risk'] = 'Medium'
            
        # Impact on capacity
        if clustering_stats['clustering_risk'] == 'High':
            clustering_stats['capacity_reduction'] = 0.5  # 50% reduction
        elif clustering_stats['clustering_risk'] == 'Medium':
            clustering_stats['capacity_reduction'] = 0.25  # 25% reduction
        else:
            clustering_stats['capacity_reduction'] = 0.0
            
        return clustering_stats
    
    def create_capacity_report(self, output_dir: Path):
        """Create comprehensive capacity analysis report"""
        
        output_dir.mkdir(exist_ok=True)
        
        logger.info("Creating capacity analysis report...")
        
        # 1. Main capacity analysis
        capacity_results = self.analyze_capacity()
        
        # 2. Session analysis
        session_results = self.analyze_session_liquidity()
        
        # 3. Trade clustering
        clustering_results = self.analyze_trade_clustering()
        
        # 4. Slippage curve
        slippage_curve = calculate_slippage_curve(
            self.pair,
            [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2],
            self.adv_usd
        )
        
        # Find maximum viable size
        viable_sizes = capacity_results[capacity_results['viable']]
        if len(viable_sizes) > 0:
            max_viable_size = viable_sizes['trade_size_mm'].max()
            max_viable_multiple = viable_sizes['size_multiple'].max()
        else:
            max_viable_size = 0
            max_viable_multiple = 0
            
        # Apply clustering adjustment
        if clustering_results.get('capacity_reduction', 0) > 0:
            adjusted_max_size = max_viable_size * (1 - clustering_results['capacity_reduction'])
            logger.info(f"Adjusting capacity for clustering: {max_viable_size:.1f}M -> {adjusted_max_size:.1f}M")
            max_viable_size = adjusted_max_size
        
        # Create visualizations
        self._create_capacity_plots(
            capacity_results, session_results, slippage_curve,
            clustering_results, output_dir
        )
        
        # Summary report
        summary = {
            'pair': self.pair,
            'adv_usd': self.adv_usd,
            'adv_billions': self.adv_usd / 1e9,
            'base_sharpe': capacity_results.iloc[0]['base_sharpe'],
            'max_viable_size_mm': max_viable_size,
            'max_viable_multiple_adv': max_viable_multiple,
            'clustering_risk': clustering_results.get('clustering_risk', 'Unknown'),
            'capacity_reduction_pct': clustering_results.get('capacity_reduction', 0) * 100,
            'best_session': session_results.loc[session_results['sharpe'].idxmax()]['session'] if len(session_results) > 0 else 'Unknown',
            'session_analysis': session_results.to_dict('records'),
            'capacity_curve': capacity_results.to_dict('records'),
            'clustering_stats': clustering_results
        }
        
        # Save results
        with open(output_dir / 'capacity_analysis.json', 'w') as f:
            json.dump(summary, f, indent=2)
            
        # Save CSV for easy analysis
        capacity_results.to_csv(output_dir / 'capacity_curve.csv', index=False)
        session_results.to_csv(output_dir / 'session_analysis.csv', index=False)
        
        logger.info(f"Capacity report saved to {output_dir}")
        
        return summary
    
    def _create_capacity_plots(self, capacity_df, session_df, slippage_df,
                              clustering, output_dir):
        """Create capacity analysis visualizations"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Sharpe degradation curve
        ax1 = axes[0, 0]
        ax1.plot(capacity_df['trade_size_mm'], capacity_df['adjusted_sharpe'], 
                'b-', marker='o', linewidth=2, markersize=8)
        ax1.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Target Sharpe = 1.0')
        ax1.axhline(y=capacity_df.iloc[0]['base_sharpe'], color='green', 
                   linestyle=':', alpha=0.5, label=f"Base Sharpe = {capacity_df.iloc[0]['base_sharpe']:.2f}")
        
        # Mark maximum viable size
        viable = capacity_df[capacity_df['viable']]
        if len(viable) > 0:
            max_viable = viable.iloc[-1]
            ax1.scatter([max_viable['trade_size_mm']], [max_viable['adjusted_sharpe']], 
                       color='red', s=100, zorder=5)
            ax1.annotate(f"Max: ${max_viable['trade_size_mm']:.1f}M", 
                        (max_viable['trade_size_mm'], max_viable['adjusted_sharpe']),
                        xytext=(10, 10), textcoords='offset points')
        
        ax1.set_xlabel('Trade Size ($M)')
        ax1.set_ylabel('Sharpe Ratio')
        ax1.set_title(f'{self.pair} - Sharpe Ratio vs Trade Size')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 2. Cost breakdown
        ax2 = axes[0, 1]
        x = capacity_df['trade_size_mm']
        ax2.fill_between(x, 0, capacity_df['spread_cost_bps'], 
                        alpha=0.5, label='Spread Cost', color='blue')
        ax2.fill_between(x, capacity_df['spread_cost_bps'], 
                        capacity_df['total_cost_bps'], 
                        alpha=0.5, label='Market Impact', color='red')
        ax2.set_xlabel('Trade Size ($M)')
        ax2.set_ylabel('Cost (bps)')
        ax2.set_title('Transaction Cost Breakdown')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Session analysis
        if len(session_df) > 0:
            ax3 = axes[1, 0]
            sessions = session_df['session']
            sharpes = session_df['sharpe']
            colors = ['green' if s > 0 else 'red' for s in sharpes]
            bars = ax3.bar(sessions, sharpes, color=colors, alpha=0.7)
            
            # Add ADV percentage on secondary axis
            ax3_twin = ax3.twinx()
            ax3_twin.plot(sessions, session_df['session_adv_pct'], 
                         'ko-', markersize=8, label='% of Daily Volume')
            ax3_twin.set_ylabel('% of Daily Volume')
            ax3_twin.legend(loc='upper right')
            
            ax3.set_ylabel('Sharpe Ratio')
            ax3.set_title('Performance by Trading Session')
            ax3.grid(True, alpha=0.3, axis='y')
            
            # Rotate labels
            plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
        
        # 4. Summary statistics
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Determine capacity assessment
        base_sharpe = capacity_df.iloc[0]['base_sharpe']
        if base_sharpe > 2.0:
            capacity_assess = "EXCELLENT"
            assess_color = "green"
        elif base_sharpe > 1.5:
            capacity_assess = "GOOD"
            assess_color = "blue"
        elif base_sharpe > 1.0:
            capacity_assess = "MODERATE"
            assess_color = "orange"
        else:
            capacity_assess = "LIMITED"
            assess_color = "red"
        
        summary_text = f"""
        Capacity Analysis Summary - {self.pair}
        
        Market Liquidity:
        Average Daily Volume: ${self.adv_usd/1e9:.1f}B
        Typical Spread: {self.spread_pips:.1f} pips
        
        Strategy Capacity:
        Base Sharpe (no impact): {base_sharpe:.3f}
        Max Size @ Sharpe ≥ 1.0: ${capacity_df[capacity_df['viable']]['trade_size_mm'].max():.1f}M
        Max as % of ADV: {capacity_df[capacity_df['viable']]['size_multiple'].max()*100:.2f}%
        
        Trade Clustering:
        Risk Level: {clustering.get('clustering_risk', 'Unknown')}
        Max Trades/Day: {clustering.get('max_trades_per_day', 0)}
        Capacity Adjustment: -{clustering.get('capacity_reduction', 0)*100:.0f}%
        
        Capacity Assessment: {capacity_assess}
        """
        
        ax4.text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center',
                fontfamily='monospace')
        
        # Add colored box for assessment
        from matplotlib.patches import Rectangle
        rect = Rectangle((0.05, 0.85), 0.9, 0.1, 
                        facecolor=assess_color, alpha=0.2)
        ax4.add_patch(rect)
        ax4.text(0.5, 0.9, capacity_assess, fontsize=16, weight='bold',
                ha='center', va='center', color=assess_color)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'capacity_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    # Example usage
    data_path = Path('../../data/AUDUSD_MASTER_15M.csv')
    data = pd.read_csv(data_path, parse_dates=['DateTime'], index_col='DateTime')
    
    # Use recent data
    data = data[-100000:]
    
    analyzer = CapacityAnalyzer(data, 'AUDUSD')
    summary = analyzer.create_capacity_report(Path('capacity_output'))
    
    print("\nCapacity Analysis Complete!")
    print(f"Maximum viable size: ${summary['max_viable_size_mm']:.1f}M")
    print(f"As % of ADV: {summary['max_viable_multiple_adv']*100:.3f}%")