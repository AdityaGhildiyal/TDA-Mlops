"""
tda_detect
==========
Topological Data Analysis for Production-Grade Anomaly Detection.

Public API
----------
    from tda_detect import TDAFeatureExtractor
    from tda_detect import TDAAnomalyDetector
    from tda_detect import TopologicalDriftDetector
"""

from tda_detect.features import TDAFeatureExtractor
from tda_detect.model   import TDAAnomalyDetector
from tda_detect.drift   import TopologicalDriftDetector

__version__ = "0.4.0"
__all__ = [
    "TDAFeatureExtractor",
    "TDAAnomalyDetector",
    "TopologicalDriftDetector",
]