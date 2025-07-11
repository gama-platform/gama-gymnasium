"""
Logging utilities for GAMA-Gymnasium

This module provides consistent logging configuration across the package.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Get a configured logger instance.
    
    Args:
        name (str): Logger name (usually __name__)
        level (int): Logging level
        
    Returns:
        logging.Logger: Configured logger
    """
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(console_handler)
        logger.setLevel(level)
    
    return logger


def setup_file_logging(
    logger: logging.Logger,
    log_file: str,
    level: int = logging.DEBUG
) -> None:
    """
    Add file logging to a logger.
    
    Args:
        logger (logging.Logger): Logger to configure
        log_file (str): Path to log file
        level (int): Logging level for file
    """
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(level)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)


def configure_root_logger(level: int = logging.INFO) -> None:
    """
    Configure the root logger for the package.
    
    Args:
        level (int): Logging level
    """
    root_logger = logging.getLogger('gama_gymnasium')
    root_logger.setLevel(level)
    
    if not root_logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        
        root_logger.addHandler(console_handler)
