# Legacy — archivos de versiones anteriores del pipeline

Esta carpeta contiene archivos de versiones anteriores del pipeline que ya
no son parte del estado actual. Se conservan como referencia historica.

**No son parte del pipeline actual y pueden referenciar codigo, targets,
feature sets o presets que ya no existen** (`_MENT14D`, `recommended`,
`rf_top30`, `core+ses+aces`, `ace_score`, `use_ace`, multi-target, etc.).

## Contenido

- `brfss_dataset_model__MENT14D.py` — wrapper del target `_MENT14D`
  (eliminado; el pipeline ahora usa solo `ADDEPEV3`).
- `tools/audit.py` — auditoria de features y comparacion de 4 targets
  candidatos (`ADDEPEV3`, `_MENT14D`, `MENTHLTH_b14`, `LSATISFY_dissat`)
  con el sistema de tiers y el set `recommended` original. Documenta por
  que se eligio `ADDEPEV3` y como se razono el set original.
- `metadata/audit/` — salidas de `tools/audit.py`:
  `feature_audit.{csv,md}` (300 variables con tier + RF + univariate + MI),
  `target_comparison.{csv,md}` (4 targets: prevalence, AUC, F1).
- `metadata/model_results/old_results/` — resultados historicos
  (`_MENT14D` + `ADDEPEV3` con el set `recommended`).
- `metadata/model_results/addepev3_30/` y `addepev3_30.zip` — resultados
  del preset `rf_top30` (eliminado; reemplazado por `default`).
- `reports/benchmark_2x2x2.md` — benchmark historico (2 targets x 2
  feature sets x 2 estados de `--include-race`).
- `reports/balancing_comparison.md` — comparacion de 4 estrategias de
  balanceo sobre la mejor combo del 2x2x2.

## Estado actual

El pipeline actual vive en la raiz del repo: target unico `ADDEPEV3`,
preset unico `default` (30 predictores seleccionados del universo completo
via `tools/select_rf_top30.py`), 5 modelos base, threshold tuning. Ver
`README.md` y `reports/changes_and_results.md` en la raiz para el estado
actual y el registro de cambios (incluye que se probo y descarto).