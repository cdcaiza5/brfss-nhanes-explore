# -*- coding: utf-8 -*-
"""BRFSS 2024 — Prediccion de _MENT14D (14+ dias de mala salud mental) con el set "recommended".

Wrapper delgado sobre brfss_dataset_model.main() con --target=_MENT14D y
--feature-set=recommended como defaults. Si el usuario pasa --target o
--feature-set en la linea de comandos, sus valores tienen precedencia.
Logica real en brfss_dataset_model.py.

Uso:
    python brfss_dataset_model__MENT14D.py --subsample 100000 --cv 5
    python brfss_dataset_model__MENT14D.py --target ADDEPEV3 --cv 3
"""
import sys

# Defaults del wrapper; se prependen solo si el usuario no los paso.
# argparse usa la ultima ocurrencia de cada flag, asi que los valores del
# usuario (si estan) quedan al final y ganan.
_DEFAULTS = ("--target", "_MENT14D", "--feature-set", "recommended")
existing = sys.argv[1:]
prefix: list[str] = []
for i in range(0, len(_DEFAULTS), 2):
    flag = _DEFAULTS[i]
    if flag not in existing:
        prefix += [flag, _DEFAULTS[i + 1]]
sys.argv = [sys.argv[0]] + prefix + existing

from brfss_dataset_model import main  # noqa: E402

main()
