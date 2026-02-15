# IFRS 9 / CPC 4966 — API Reference

## Visão Geral

Módulos para compliance com IFRS 9 / CPC 4966:

- **Stress Testing** (`core.stress_testing`) — choques e cenários sobre matrizes de transição
- **Staging** (`core.staging`) — classificação em Stage 1 / 2 / 3
- **ECL Calculator** (`core.ecl_calculator`) — Expected Credit Loss por estágio

---

## Stress Testing

### `apply_scalar_stress()`

Multiplica probabilidades de migração off-diagonal por um fator escalar, renormalizando cada linha.

```python
apply_scalar_stress(
    transition_matrix: pd.DataFrame,
    stress_factor: float = 1.5,
    direction: str = 'worsen'
) -> pd.DataFrame
```

**Parâmetros:**

| Parâmetro | Tipo | Default | Descrição |
|-----------|------|---------|-----------|
| `transition_matrix` | pd.DataFrame | — | Matriz de transição base |
| `stress_factor` | float | 1.5 | Fator multiplicador (> 0) |
| `direction` | str | 'worsen' | `'worsen'` (piora), `'improve'` (melhora), `'both'` (ambos) |

**Retorno:** pd.DataFrame — Matriz estressada (linhas somam 1.0)

**Raises:** ValueError se `stress_factor < 0` ou `direction` inválido.

**Exemplo:**
```python
from credit_risk_transition_matrix import apply_scalar_stress

# Piora: migração para buckets piores x1.5
stressed = apply_scalar_stress(learner.transition_matrix_, stress_factor=1.5)

# Melhora: migração para buckets melhores x0.7
improved = apply_scalar_stress(learner.transition_matrix_, stress_factor=0.7, direction='improve')
```

---

### `apply_z_shift()`

Desloca massa de probabilidade em direção a buckets piores (z > 0) ou melhores (z < 0).

```python
apply_z_shift(
    transition_matrix: pd.DataFrame,
    z_shift: float = 0.5
) -> pd.DataFrame
```

**Parâmetros:**

| Parâmetro | Tipo | Default | Descrição |
|-----------|------|---------|-----------|
| `transition_matrix` | pd.DataFrame | — | Matriz de transição base |
| `z_shift` | float | 0.5 | Magnitude do shift (positivo = piora, negativo = melhora) |

**Retorno:** pd.DataFrame — Matriz com shift aplicado.

---

### `generate_scenarios()`

Gera múltiplas matrizes estressadas para análise de cenários.

```python
generate_scenarios(
    transition_matrix: pd.DataFrame,
    scenarios_config: Optional[Dict[str, Dict[str, float]]] = None
) -> Dict[str, pd.DataFrame]
```

**Parâmetros:**

| Parâmetro | Tipo | Default | Descrição |
|-----------|------|---------|-----------|
| `transition_matrix` | pd.DataFrame | — | Matriz base (TTC) |
| `scenarios_config` | Dict | None | Config de cenários (veja abaixo) |

**Config padrão:**
```python
{
    'base':        {'factor': 1.0, 'weight': 0.50},
    'optimistic':  {'factor': 0.7, 'weight': 0.20},
    'pessimistic': {'factor': 1.5, 'weight': 0.30}
}
```

**Retorno:** Dict com chaves de cenário → pd.DataFrame, mais `'_weights'` → Dict[str, float].

**Raises:** ValueError se os pesos não somarem 1.0 (tolerância 1e-6).

**Exemplo:**
```python
from credit_risk_transition_matrix import generate_scenarios

scenarios = generate_scenarios(learner.transition_matrix_, {
    'base':        {'factor': 1.0, 'weight': 0.40},
    'otimista':    {'factor': 0.7, 'weight': 0.30},
    'pessimista':  {'factor': 1.8, 'weight': 0.30}
})

for name, matrix in scenarios.items():
    if name != '_weights':
        print(f"{name}: PD bucket '0-14' -> '360+' = {matrix.iloc[0, -1]:.4f}")
```

---

### `weighted_scenario_matrix()`

Combina matrizes de cenário em uma única matriz forward-looking ponderada.

```python
weighted_scenario_matrix(
    scenario_matrices: Dict[str, pd.DataFrame],
    weights: Optional[Dict[str, float]] = None
) -> pd.DataFrame
```

**Parâmetros:**

| Parâmetro | Tipo | Default | Descrição |
|-----------|------|---------|-----------|
| `scenario_matrices` | Dict | — | Output de `generate_scenarios()` ou dict manual |
| `weights` | Dict | None | Pesos por cenário; se None, usa `'_weights'` do dict |

**Retorno:** pd.DataFrame — M_weighted = Σ(w_i × M_i)

**Exemplo:**
```python
from credit_risk_transition_matrix import weighted_scenario_matrix

fl_matrix = weighted_scenario_matrix(scenarios)
```

---

## Staging

### `classify_stage()`

Classifica cada observação em IFRS 9 Stage 1, 2 ou 3.

```python
classify_stage(
    df: pd.DataFrame,
    bucket_col: str = 'bucket',
    bucket_labels: Optional[List[str]] = None,
    stage1_max_bucket: Optional[str] = None,
    default_buckets: Optional[List[str]] = None,
    origin_bucket_col: Optional[str] = None,
    deterioration_threshold: int = 2
) -> pd.DataFrame
```

**Parâmetros:**

| Parâmetro | Tipo | Default | Descrição |
|-----------|------|---------|-----------|
| `df` | pd.DataFrame | — | DataFrame com coluna de bucket |
| `bucket_col` | str | 'bucket' | Coluna com bucket/grau de risco |
| `bucket_labels` | List[str] | None | Lista ordenada de buckets (melhor→pior). Inferida se None |
| `stage1_max_bucket` | str | None | Bucket máximo para Stage 1. Default: 2º bucket |
| `default_buckets` | List[str] | None | Buckets de default (Stage 3). Default: último bucket |
| `origin_bucket_col` | str | None | Coluna com bucket na originação (para teste de deterioração) |
| `deterioration_threshold` | int | 2 | Posições de piora para trigger de Stage 2 |

**Retorno:** pd.DataFrame — Cópia com coluna `'stage'` (valores: 1, 2, 3).

**Lógica de classificação:**
1. **Stage 3** — bucket está em `default_buckets`
2. **Stage 1** — bucket ≤ `stage1_max_bucket` E não deteriorou ≥ threshold
3. **Stage 2** — todos os demais

**Exemplo:**
```python
from credit_risk_transition_matrix import classify_stage

df_staged = classify_stage(
    df,
    bucket_col='bucket',
    bucket_labels=learner.bucket_labels_,
    stage1_max_bucket='15-29',
    default_buckets=['360+']
)
print(df_staged['stage'].value_counts())
```

---

### `stage_migration_matrix()`

Calcula a matriz 3×3 de migração entre estágios IFRS 9.

```python
stage_migration_matrix(
    df: pd.DataFrame,
    id_col: str = 'id_contrato',
    time_col: str = 'data_ref',
    stage_col: str = 'stage'
) -> pd.DataFrame
```

**Parâmetros:**

| Parâmetro | Tipo | Default | Descrição |
|-----------|------|---------|-----------|
| `df` | pd.DataFrame | — | DataFrame com classificação de stage |
| `id_col` | str | 'id_contrato' | Coluna de identificador |
| `time_col` | str | 'data_ref' | Coluna de data |
| `stage_col` | str | 'stage' | Coluna de estágio |

**Retorno:** pd.DataFrame — Matriz 3×3 (linhas somam 1.0)

---

### `stage_summary()`

Resumo do portfólio por estágio IFRS 9.

```python
stage_summary(
    df: pd.DataFrame,
    stage_col: str = 'stage',
    ead_col: Optional[str] = None,
    pd_col: Optional[str] = None
) -> pd.DataFrame
```

**Parâmetros:**

| Parâmetro | Tipo | Default | Descrição |
|-----------|------|---------|-----------|
| `df` | pd.DataFrame | — | DataFrame com classificação de stage |
| `stage_col` | str | 'stage' | Coluna de estágio |
| `ead_col` | str | None | Coluna de EAD (opcional) |
| `pd_col` | str | None | Coluna de PD (opcional) |

**Colunas retornadas:** `stage`, `count`, `pct_count`, e opcionalmente `ead_total`, `ead_mean`, `pct_ead`, `pd_mean`, `pd_median`.

---

## ECL Calculator

### `calculate_ecl()`

Calcula ECL = PD × LGD × EAD com desconto opcional.

```python
calculate_ecl(
    pd_values: pd.Series,
    ead_values: pd.Series,
    lgd: Union[float, pd.Series] = 0.45,
    discount_rate: float = 0.0
) -> pd.Series
```

**Parâmetros:**

| Parâmetro | Tipo | Default | Descrição |
|-----------|------|---------|-----------|
| `pd_values` | pd.Series | — | PD por observação |
| `ead_values` | pd.Series | — | EAD por observação |
| `lgd` | float/pd.Series | 0.45 | LGD fixo ou por observação |
| `discount_rate` | float | 0.0 | Taxa efetiva anual de desconto |

**Retorno:** pd.Series — ECL por observação.

---

### `ecl_by_stage()`

Calcula ECL por contrato usando PD com horizonte adequado ao estágio:

- **Stage 1:** PD(12 meses) × LGD × EAD
- **Stage 2:** PD(lifetime) × LGD × EAD
- **Stage 3:** 1.0 × LGD × EAD

```python
ecl_by_stage(
    transition_matrix: pd.DataFrame,
    df: pd.DataFrame,
    bucket_col: str = 'bucket',
    stage_col: str = 'stage',
    ead_col: str = 'ead',
    lgd: Union[float, str] = 0.45,
    n_periods_12m: int = 12,
    n_periods_lifetime: int = 60,
    default_states: Optional[List[str]] = None,
    discount_rate: float = 0.0
) -> pd.DataFrame
```

**Parâmetros:**

| Parâmetro | Tipo | Default | Descrição |
|-----------|------|---------|-----------|
| `transition_matrix` | pd.DataFrame | — | Matriz de transição (ou ponderada por cenário) |
| `df` | pd.DataFrame | — | DataFrame com bucket, stage e EAD |
| `bucket_col` | str | 'bucket' | Coluna de bucket |
| `stage_col` | str | 'stage' | Coluna de estágio |
| `ead_col` | str | 'ead' | Coluna de EAD |
| `lgd` | float/str | 0.45 | LGD fixo ou nome de coluna no df |
| `n_periods_12m` | int | 12 | Períodos para PD 12 meses (Stage 1) |
| `n_periods_lifetime` | int | 60 | Períodos para PD lifetime (Stage 2) |
| `default_states` | List[str] | None | Buckets de default (default: último) |
| `discount_rate` | float | 0.0 | Taxa anual de desconto |

**Retorno:** pd.DataFrame — Cópia com colunas `'pd_used'`, `'lgd_used'`, `'ecl'`.

**Exemplo:**
```python
from credit_risk_transition_matrix import ecl_by_stage

df_ecl = ecl_by_stage(
    transition_matrix=learner.transition_matrix_,
    df=df_staged,
    bucket_col='bucket',
    stage_col='stage',
    ead_col='ead',
    lgd=0.45
)
print(f"ECL Total: {df_ecl['ecl'].sum():,.2f}")
```

---

### `portfolio_ecl()`

Agrega ECL no nível do portfólio com breakdown por estágio e segmento.

```python
portfolio_ecl(
    ecl_df: pd.DataFrame,
    stage_col: str = 'stage',
    ead_col: str = 'ead',
    ecl_col: str = 'ecl',
    segment_col: Optional[str] = None
) -> pd.DataFrame
```

**Parâmetros:**

| Parâmetro | Tipo | Default | Descrição |
|-----------|------|---------|-----------|
| `ecl_df` | pd.DataFrame | — | Output de `ecl_by_stage()` |
| `stage_col` | str | 'stage' | Coluna de estágio |
| `ead_col` | str | 'ead' | Coluna de EAD |
| `ecl_col` | str | 'ecl' | Coluna de ECL |
| `segment_col` | str | None | Coluna de segmento (opcional) |

**Colunas retornadas:** `stage`, `count`, `ead_total`, `ecl_total`, `pd_mean`, `coverage_ratio` — incluindo linha `'Total'`.

**Exemplo:**
```python
from credit_risk_transition_matrix import portfolio_ecl

resumo = portfolio_ecl(df_ecl, segment_col='segmento')
print(resumo.to_string(index=False))
```

---

## Pipeline completo IFRS 9

```python
from credit_risk_transition_matrix import (
    TransitionMatrixLearner,
    apply_scalar_stress, generate_scenarios, weighted_scenario_matrix,
    classify_stage, stage_summary,
    ecl_by_stage, portfolio_ecl
)

# 1) Treinar matriz
learner = TransitionMatrixLearner(buckets=[0,15,30,60,90,120,180,240,360])
learner.fit(df_train, id_col='id_contrato', time_col='data_ref', bucket_col='dias_atraso')

# 2) Cenários de stress
scenarios = generate_scenarios(learner.transition_matrix_, {
    'base':       {'factor': 1.0, 'weight': 0.40},
    'otimista':   {'factor': 0.7, 'weight': 0.30},
    'pessimista': {'factor': 1.8, 'weight': 0.30}
})
fl_matrix = weighted_scenario_matrix(scenarios)

# 3) Staging
snapshot = df_train.groupby('id_contrato').last().reset_index()
snapshot['bucket'] = snapshot['dias_atraso'].apply(
    lambda d: learner.bucket_labels_[min(d // 15, len(learner.bucket_labels_)-1)]
)

df_staged = classify_stage(
    snapshot,
    bucket_col='bucket',
    bucket_labels=learner.bucket_labels_,
    stage1_max_bucket='15-29',
    default_buckets=['360+']
)

# 4) ECL
df_ecl = ecl_by_stage(fl_matrix, df_staged, lgd=0.45)
resumo = portfolio_ecl(df_ecl, segment_col='segmento')
print(resumo)
```

## Veja Também

- [TransitionMatrixLearner API](transition_matrix_learner.md)
- [Risk Metrics](risk_metrics.md)
- [PD Calculations](../theory/pd_calculations.md)
