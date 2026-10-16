"""
limiter.py — single shared slowapi Limiter (v3.7).

main.py and routers/query.py each used to create their own Limiter, so the
instance registered on app.state was not the one enforcing the query limits.
Both now import this one object.
"""

from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
