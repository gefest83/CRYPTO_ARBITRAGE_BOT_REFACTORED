"""Isolated research package.

Never imported by production trading code (triangle, transfer, execution,
risk, telegram, kronos).  All research modules are read-only and never
place orders or touch wallets.
"""
