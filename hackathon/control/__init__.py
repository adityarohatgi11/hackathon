"""Control module for GridPilot-GT cooling and system constraints."""

from .cooling_model import cooling_for_gpu_kW

__all__ = ['cooling_for_gpu_kW'] 