"""Loss functions for adversarial training (TRADES, AMS distillation)."""

from src.losses.ams import ams_distill_loss
from src.losses.trades import trades_loss

__all__ = ["ams_distill_loss", "trades_loss"]
