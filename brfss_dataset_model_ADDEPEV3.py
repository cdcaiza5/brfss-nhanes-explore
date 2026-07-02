# -*- coding: utf-8 -*-
"""Wrapper delgado: ADDEPEV3 (depresion de por vida) con el set "recommended".

Logica real en brfss_dataset_model.py.
"""
import sys
from brfss_dataset_model import main

main(
    defaults=["--target", "ADDEPEV3", "--feature-set", "recommended"],
    user_argv=sys.argv[1:],
)
