"""
QuantLab - Research Framework for Systematic Trading
"""

from .signals import momentum, ma_crossover
from .portfolio import Backtest
from .costs import FXCosts

__version__ = "1.0.0"
__all__ = ["momentum", "ma_crossover", "Backtest", "FXCosts"]