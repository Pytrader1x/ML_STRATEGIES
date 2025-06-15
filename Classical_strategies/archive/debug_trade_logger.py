"""
Trade Debug Logger - Comprehensive JSON logging for all trade lifecycle events
"""

import json
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import numpy as np


@dataclass
class TradeDebugEntry:
    """Complete trade lifecycle tracking"""
    # Trade identification
    trade_id: int
    entry_time: str
    
    # Entry details
    entry_price: float
    direction: str  # 'long' or 'short'
    initial_size_millions: float
    confidence: float
    is_relaxed: bool
    entry_logic: str
    
    # Risk management setup
    sl_price: float
    sl_distance_pips: float
    tp1_price: float
    tp2_price: float
    tp3_price: float
    
    # TP1 tracking
    tp1_hit: bool = False
    tp1_hit_time: Optional[str] = None
    tp1_exit_price: Optional[float] = None
    tp1_size_closed_millions: float = 0.0
    tp1_pnl_dollars: float = 0.0
    tp1_pips_captured: float = 0.0
    position_after_tp1_millions: float = 0.0
    
    # TP2 tracking
    tp2_hit: bool = False
    tp2_hit_time: Optional[str] = None
    tp2_exit_price: Optional[float] = None
    tp2_size_closed_millions: float = 0.0
    tp2_pnl_dollars: float = 0.0
    tp2_pips_captured: float = 0.0
    position_after_tp2_millions: float = 0.0
    
    # TP3 tracking
    tp3_hit: bool = False
    tp3_hit_time: Optional[str] = None
    tp3_exit_price: Optional[float] = None
    tp3_size_closed_millions: float = 0.0
    tp3_pnl_dollars: float = 0.0
    tp3_pips_captured: float = 0.0
    position_after_tp3_millions: float = 0.0
    
    # PPT (Partial Profit Taking) tracking
    ppt_active: bool = False
    ppt_trigger_price: Optional[float] = None
    ppt_trigger_time: Optional[str] = None
    ppt_size_closed_millions: float = 0.0
    ppt_pnl_dollars: float = 0.0
    ppt_pips_captured: float = 0.0
    position_after_ppt_millions: float = 0.0
    
    # Final exit tracking
    exit_time: Optional[str] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None  # 'stop_loss', 'trailing_stop', 'take_profit_3', etc.
    final_position_size_millions: float = 0.0
    final_exit_pnl_dollars: float = 0.0
    final_exit_pips: float = 0.0
    
    # Trade summary
    total_pnl_dollars: float = 0.0
    total_pips_captured: float = 0.0
    trade_duration_hours: float = 0.0
    max_favorable_excursion_pips: float = 0.0
    max_adverse_excursion_pips: float = 0.0
    
    # Validation flags
    pnl_components_match: bool = True
    size_tracking_valid: bool = True
    validation_errors: List[str] = None
    
    def __post_init__(self):
        if self.validation_errors is None:
            self.validation_errors = []
        
        # Initialize position tracking
        self.position_after_ppt_millions = self.initial_size_millions
        self.position_after_tp1_millions = self.initial_size_millions
        self.position_after_tp2_millions = self.initial_size_millions
        self.position_after_tp3_millions = self.initial_size_millions
    
    def validate(self):
        """Validate trade consistency"""
        errors = []
        
        # Check size tracking
        total_closed = (self.ppt_size_closed_millions + 
                       self.tp1_size_closed_millions + 
                       self.tp2_size_closed_millions + 
                       self.tp3_size_closed_millions +
                       self.final_position_size_millions)
        
        if abs(total_closed - self.initial_size_millions) > 0.001:
            errors.append(f"Size mismatch: closed {total_closed:.3f}M vs initial {self.initial_size_millions:.3f}M")
            self.size_tracking_valid = False
        
        # Check PnL components
        component_pnl = (self.ppt_pnl_dollars + 
                        self.tp1_pnl_dollars + 
                        self.tp2_pnl_dollars + 
                        self.tp3_pnl_dollars +
                        self.final_exit_pnl_dollars)
        
        if abs(component_pnl - self.total_pnl_dollars) > 1.0:
            errors.append(f"PnL mismatch: components sum to ${component_pnl:.2f} vs total ${self.total_pnl_dollars:.2f}")
            self.pnl_components_match = False
        
        # Check TP hit sequence
        if self.tp3_hit and not self.tp2_hit:
            errors.append("TP3 hit without TP2 being hit")
        if self.tp2_hit and not self.tp1_hit:
            errors.append("TP2 hit without TP1 being hit")
        
        self.validation_errors = errors
        return len(errors) == 0


class TradeDebugLogger:
    def __init__(self, output_dir: str = "debug"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.trades: Dict[int, TradeDebugEntry] = {}
        self.current_trade_id = 1
        
    def create_trade(self, entry_time: pd.Timestamp, entry_price: float, 
                    direction: str, size_millions: float, confidence: float,
                    is_relaxed: bool, entry_logic: str, sl_price: float,
                    tp1_price: float, tp2_price: float, tp3_price: float) -> int:
        """Create new trade entry"""
        trade_id = self.current_trade_id
        self.current_trade_id += 1
        
        # Calculate SL distance in pips
        if direction == 'long':
            sl_distance_pips = (entry_price - sl_price) * 10000
        else:
            sl_distance_pips = (sl_price - entry_price) * 10000
        
        trade = TradeDebugEntry(
            trade_id=trade_id,
            entry_time=entry_time.strftime("%Y-%m-%d %H:%M:%S"),
            entry_price=entry_price,
            direction=direction,
            initial_size_millions=size_millions,
            confidence=confidence,
            is_relaxed=is_relaxed,
            entry_logic=entry_logic,
            sl_price=sl_price,
            sl_distance_pips=sl_distance_pips,
            tp1_price=tp1_price,
            tp2_price=tp2_price,
            tp3_price=tp3_price
        )
        
        self.trades[trade_id] = trade
        return trade_id
    
    def update_ppt(self, trade_id: int, trigger_time: pd.Timestamp, 
                   trigger_price: float, size_closed: float, pnl: float):
        """Update PPT exit"""
        if trade_id not in self.trades:
            return
        
        trade = self.trades[trade_id]
        trade.ppt_active = True
        trade.ppt_trigger_time = trigger_time.strftime("%Y-%m-%d %H:%M:%S")
        trade.ppt_trigger_price = trigger_price
        trade.ppt_size_closed_millions = size_closed / 1_000_000
        trade.ppt_pnl_dollars = pnl
        
        # Calculate pips
        if trade.direction == 'long':
            trade.ppt_pips_captured = (trigger_price - trade.entry_price) * 10000
        else:
            trade.ppt_pips_captured = (trade.entry_price - trigger_price) * 10000
        
        # Update remaining position
        trade.position_after_ppt_millions = trade.initial_size_millions - trade.ppt_size_closed_millions
    
    def update_tp_hit(self, trade_id: int, tp_level: int, hit_time: pd.Timestamp,
                      exit_price: float, size_closed: float, pnl: float):
        """Update TP hit"""
        if trade_id not in self.trades:
            return
        
        trade = self.trades[trade_id]
        size_millions = size_closed / 1_000_000
        
        # Calculate pips
        if trade.direction == 'long':
            pips = (exit_price - trade.entry_price) * 10000
        else:
            pips = (trade.entry_price - exit_price) * 10000
        
        if tp_level == 1:
            trade.tp1_hit = True
            trade.tp1_hit_time = hit_time.strftime("%Y-%m-%d %H:%M:%S")
            trade.tp1_exit_price = exit_price
            trade.tp1_size_closed_millions = size_millions
            trade.tp1_pnl_dollars = pnl
            trade.tp1_pips_captured = pips
            # Account for PPT if active
            base_position = trade.position_after_ppt_millions if trade.ppt_active else trade.initial_size_millions
            trade.position_after_tp1_millions = base_position - size_millions
            
        elif tp_level == 2:
            trade.tp2_hit = True
            trade.tp2_hit_time = hit_time.strftime("%Y-%m-%d %H:%M:%S")
            trade.tp2_exit_price = exit_price
            trade.tp2_size_closed_millions = size_millions
            trade.tp2_pnl_dollars = pnl
            trade.tp2_pips_captured = pips
            trade.position_after_tp2_millions = trade.position_after_tp1_millions - size_millions
            
        elif tp_level == 3:
            trade.tp3_hit = True
            trade.tp3_hit_time = hit_time.strftime("%Y-%m-%d %H:%M:%S")
            trade.tp3_exit_price = exit_price
            trade.tp3_size_closed_millions = size_millions
            trade.tp3_pnl_dollars = pnl
            trade.tp3_pips_captured = pips
            trade.position_after_tp3_millions = trade.position_after_tp2_millions - size_millions
    
    def update_final_exit(self, trade_id: int, exit_time: pd.Timestamp,
                         exit_price: float, exit_reason: str, remaining_size: float,
                         final_pnl: float, total_trade_pnl: float):
        """Update final exit"""
        if trade_id not in self.trades:
            return
        
        trade = self.trades[trade_id]
        trade.exit_time = exit_time.strftime("%Y-%m-%d %H:%M:%S")
        trade.exit_price = exit_price
        trade.exit_reason = exit_reason
        trade.final_position_size_millions = remaining_size / 1_000_000
        trade.final_exit_pnl_dollars = final_pnl
        trade.total_pnl_dollars = total_trade_pnl
        
        # Calculate final exit pips
        if trade.direction == 'long':
            trade.final_exit_pips = (exit_price - trade.entry_price) * 10000
        else:
            trade.final_exit_pips = (trade.entry_price - exit_price) * 10000
        
        # Calculate total pips (weighted average)
        total_size = (trade.ppt_size_closed_millions + trade.tp1_size_closed_millions + 
                     trade.tp2_size_closed_millions + trade.tp3_size_closed_millions + 
                     trade.final_position_size_millions)
        
        if total_size > 0:
            trade.total_pips_captured = (
                (trade.ppt_pips_captured * trade.ppt_size_closed_millions +
                 trade.tp1_pips_captured * trade.tp1_size_closed_millions +
                 trade.tp2_pips_captured * trade.tp2_size_closed_millions +
                 trade.tp3_pips_captured * trade.tp3_size_closed_millions +
                 trade.final_exit_pips * trade.final_position_size_millions) / total_size
            )
        
        # Calculate trade duration
        entry_dt = pd.Timestamp(trade.entry_time)
        exit_dt = pd.Timestamp(trade.exit_time)
        trade.trade_duration_hours = (exit_dt - entry_dt).total_seconds() / 3600
        
        # Validate the trade
        trade.validate()
    
    def update_excursions(self, trade_id: int, max_favorable_pips: float, max_adverse_pips: float):
        """Update maximum excursions during trade"""
        if trade_id not in self.trades:
            return
        
        trade = self.trades[trade_id]
        trade.max_favorable_excursion_pips = max_favorable_pips
        trade.max_adverse_excursion_pips = max_adverse_pips
    
    def save_debug_log(self, timestamp: str):
        """Save debug log to JSON file"""
        filename = self.output_dir / f"trade_debug_log_{timestamp}.json"
        
        # Convert to serializable format
        debug_data = {
            "timestamp": timestamp,
            "total_trades": len(self.trades),
            "trades": {}
        }
        
        for trade_id, trade in self.trades.items():
            trade_dict = asdict(trade)
            # Ensure all values are JSON serializable
            for key, value in trade_dict.items():
                if isinstance(value, (np.integer, np.floating)):
                    trade_dict[key] = float(value)
                elif isinstance(value, np.ndarray):
                    trade_dict[key] = value.tolist()
            
            debug_data["trades"][str(trade_id)] = trade_dict
        
        # Add summary statistics
        debug_data["summary"] = self._calculate_summary()
        
        with open(filename, 'w') as f:
            json.dump(debug_data, f, indent=2)
        
        print(f"Debug log saved to: {filename}")
        
        # Also save a CSV version for easy analysis
        self._save_csv_summary(timestamp)
    
    def _calculate_summary(self) -> Dict[str, Any]:
        """Calculate summary statistics"""
        valid_trades = sum(1 for t in self.trades.values() if len(t.validation_errors) == 0)
        invalid_trades = len(self.trades) - valid_trades
        
        total_pnl = sum(t.total_pnl_dollars for t in self.trades.values())
        winning_trades = sum(1 for t in self.trades.values() if t.total_pnl_dollars > 0)
        losing_trades = sum(1 for t in self.trades.values() if t.total_pnl_dollars < 0)
        
        tp1_hits = sum(1 for t in self.trades.values() if t.tp1_hit)
        tp2_hits = sum(1 for t in self.trades.values() if t.tp2_hit)
        tp3_hits = sum(1 for t in self.trades.values() if t.tp3_hit)
        ppt_triggers = sum(1 for t in self.trades.values() if t.ppt_active)
        
        return {
            "total_trades": len(self.trades),
            "valid_trades": valid_trades,
            "invalid_trades": invalid_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": winning_trades / len(self.trades) * 100 if self.trades else 0,
            "total_pnl": total_pnl,
            "tp1_hit_rate": tp1_hits / len(self.trades) * 100 if self.trades else 0,
            "tp2_hit_rate": tp2_hits / len(self.trades) * 100 if self.trades else 0,
            "tp3_hit_rate": tp3_hits / len(self.trades) * 100 if self.trades else 0,
            "ppt_trigger_rate": ppt_triggers / len(self.trades) * 100 if self.trades else 0,
            "validation_errors": [
                {"trade_id": t.trade_id, "errors": t.validation_errors}
                for t in self.trades.values() if t.validation_errors
            ]
        }
    
    def _save_csv_summary(self, timestamp: str):
        """Save CSV summary for easy analysis"""
        rows = []
        for trade in self.trades.values():
            rows.append({
                "trade_id": trade.trade_id,
                "entry_time": trade.entry_time,
                "direction": trade.direction,
                "initial_size_m": trade.initial_size_millions,
                "confidence": trade.confidence,
                "entry_logic": trade.entry_logic,
                "ppt_pnl": trade.ppt_pnl_dollars,
                "tp1_pnl": trade.tp1_pnl_dollars,
                "tp2_pnl": trade.tp2_pnl_dollars,
                "tp3_pnl": trade.tp3_pnl_dollars,
                "final_exit_pnl": trade.final_exit_pnl_dollars,
                "total_pnl": trade.total_pnl_dollars,
                "exit_reason": trade.exit_reason,
                "duration_hours": trade.trade_duration_hours,
                "validation_errors": "; ".join(trade.validation_errors)
            })
        
        if rows:
            df = pd.DataFrame(rows)
            csv_file = self.output_dir / f"trade_debug_summary_{timestamp}.csv"
            df.to_csv(csv_file, index=False)
            print(f"CSV summary saved to: {csv_file}")


if __name__ == "__main__":
    # Example usage
    logger = TradeDebugLogger()
    
    # Create a trade
    trade_id = logger.create_trade(
        entry_time=pd.Timestamp("2023-08-01 10:00:00"),
        entry_price=1.2500,
        direction="long",
        size_millions=1.0,
        confidence=75.5,
        is_relaxed=False,
        entry_logic="Standard (NTI+MB+IC)",
        sl_price=1.2450,
        tp1_price=1.2550,
        tp2_price=1.2600,
        tp3_price=1.2650
    )
    
    # Simulate PPT
    logger.update_ppt(trade_id, pd.Timestamp("2023-08-01 10:30:00"), 1.2530, 700000, 210.0)
    
    # Simulate TP1 hit
    logger.update_tp_hit(trade_id, 1, pd.Timestamp("2023-08-01 11:00:00"), 1.2550, 300000, 150.0)
    
    # Final exit
    logger.update_final_exit(
        trade_id=trade_id,
        exit_time=pd.Timestamp("2023-08-01 12:00:00"),
        exit_price=1.2480,
        exit_reason="stop_loss",
        remaining_size=0,
        final_pnl=-60.0,
        total_trade_pnl=300.0
    )
    
    # Save debug log
    logger.save_debug_log("test_20231201")