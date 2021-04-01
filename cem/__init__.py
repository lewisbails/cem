'''A micro-library for coarsened exact matching for causal inference'''
__author__ = """Lewis Bails"""
__email__ = 'lewis.bails@gmail.com'
__version__ = '0.1.4'
from .cem import CEM, match
__all__ = ["CEM", "match"]
