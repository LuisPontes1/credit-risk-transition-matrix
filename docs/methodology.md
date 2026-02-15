# Methodology Guide: Contract Risk Engine & IFRS 9

This document details the mathematical framework and regulatory compliance logic implemented in the **Contract Risk Engine**.

---

## 1. Expected Credit Loss (ECL)

The core objective is to calculate the **Expected Credit Loss** for each contract, defined as the probability-weighted estimate of credit losses over the expected life of the financial instrument.

$$ ECL = \sum_{t=1}^{T} \frac{PD_t \times LGD_t \times EAD_t}{(1 + EIR)^t} $$

Where:
- **$PD_t$**: Marginal Probability of Default in period $t$.
- **$LGD_t$**: Loss Given Default in period $t$ (severity).
- **$EAD_t$**: Exposure at Default in period $t$ (outstanding balance).
- **$EIR$**: Effective Interest Rate (taxa de juros efetiva).
- **$T$**: Contract lifetime (or 12 months for Stage 1).

---

## 2. Probability of Default (PD)

### Transition Matrix Approach
The engine uses **Markov Chain Transition Matrices** to estimate PDs. A transition matrix $M$ describes the probability of migrating from state $i$ (e.g., 0-30 DPD) to state $j$ (e.g., Default) over a single period.

The **Cumulative PD** for $n$ periods is derived from the matrix power $M^n$:

$$ PD_{cum}(t) = P(State_t \in Default | State_0 = k) $$

### Marginal PD
For the ECL calculation, we need the conditional probability of defaulting *specifically* in period $t$, given survival up to $t-1$:

$$ PD_{marginal}(t) = PD_{cum}(t) - PD_{cum}(t-1) $$

---

## 3. Loss Given Default (LGD)

LGD represents the percentage of exposure lost if a default occurs.

$$ LGD = 1 - Recovery Rate $$

### Empirical Estimation
The `LGDRatingEngine` estimates LGD based on historical recovery data, segmented by product type and collateral.

1.  **Cure Rate**: Probability of a defaulted contract returning to performing status.
2.  **Recovery Amount**: Cash collected after default.

### Downturn Calibration
For regulatory compliance, LGD estimates can be calibrated to "downturn" conditions (periods of high defaults) rather than long-run averages, often using a scalar adjustment or specific historical windows.

---

## 4. Exposure at Default (EAD)

EAD is the predicted exposure at the time of default.

### Term Loans
For fixed-term loans (installment), EAD is the amortized balance schedule.

$$ EAD_t = Balance_{t-1} - PrincipalPayment_t $$

### Revolving Products (Credit Cards, Overdrafts)
For revolving lines, the exposure includes drawn balance plus a portion of the undrawn limit, estimated via **Credit Conversion Factor (CCF)**.

$$ EAD = Balance_{drawn} + CCF \times (Limit_{total} - Balance_{drawn}) $$

The `aad_engine.predict_ccf` method estimates CCF empirically as:

$$ CCF = \frac{EAD_{observed} - Balance_{T-12}}{Limit_{T-12} - Balance_{T-12}} $$

Bounded by [0, 1].

---

## 5. Significant Increase in Credit Risk (SICR) - Staging

Per IFRS 9 / CPC 4.966, contracts are classified into three stages:

| Stage | Criteria | ECL Horizon | Interest Income |
|-------|----------|-------------|-----------------|
| **Stage 1** | No significant deterioration | 12 Months | On Gross Balance |
| **Stage 2** | Significant increase in risk | Lifetime | On Gross Balance |
| **Stage 3** | Default / Impaired | Lifetime | On Net Balance (Post-Impairment) |

### Triggers Implemented
1.  **Relative Threshold**: $PD_{now} / PD_{origination} > 2.0$ (example).
2.  **Absolute Threshold**: $PD_{now} - PD_{origination} > 30 bps$.
3.  **Backstop**: Days Past Due (DPD) > 30 days.
4.  **Low Risk Exemption**: If $PD_{now} < 0.3\%$ (Investment Grade), force Stage 1.

---

## 6. Forward-Looking Information

ECL must reflect "reasonable and supportable information" about future economic conditions. We implement this via **Macro-Economic Scenarios**.

$$ ECL_{final} = \sum_{s \in \{base, opt, pess\}} w_s \times ECL_s $$

Where $w_s$ is the probability weight of scenario $s$.

### Overlay Application
The engine applies scalar adjustments to PD and LGD based on the scenario:
- **Optimistic**: PD Multiplier = 0.8x
- **Base**: PD Multiplier = 1.0x
- **Pessimistic**: PD Multiplier = 1.4x

These scalars can be derived from regression models linking macro variables (GDP, Unemployment) to default rates (Vasicek Model).
