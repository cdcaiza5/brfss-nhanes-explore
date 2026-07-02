# -*- coding: utf-8 -*-
"""Wrapper delgado: _MENT14D (14+ dias de mala salud mental) con el set "recommended".

Logica real en brfss_dataset_model.py.
"""
import sys
from brfss_dataset_model import main

main(
    defaults=["--target", "_MENT14D", "--feature-set", "recommended"],
    user_argv=sys.argv[1:],
)
