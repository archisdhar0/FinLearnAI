"""
Authentication module for FinLearn AI
"""

from .database import (
    signup,
    login,
    get_user,
    save_progress,
    load_progress,
    get_auth,
    get_database_backend
)

__all__ = [
    'signup',
    'login', 
    'get_user',
    'save_progress',
    'load_progress',
    'get_auth',
    'get_database_backend'
]
