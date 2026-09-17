"""
Root conftest.py — Federated Learning Network Simulator
=========================================================
The circular import between src.core and src.algorithms is now resolved
in the source __init__.py files via lazy __getattr__ imports (PEP 562).
This conftest no longer needs to pre-import modules in dependency order.

Author: Frank Meyo
Project: FL Network Simulator — UNZA CS Final Year Project 2026
"""
