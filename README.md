# Predictive Maintenance for ExxonMobil: Upstream & Downstream Operations

## BC2407 Analytics II | AY2023/24 Semester 2 | Team 1

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [ExxonMobil Operational Context](#2-exxonmobil-operational-context)
3. [Part A: Upstream Prediction — Gas Turbine Failure in Cogeneration Plants](#3-part-a-upstream-prediction--gas-turbine-failure-in-cogeneration-plants)
4. [Part B: Downstream Prediction — Refinery Process Equipment Failure](#4-part-b-downstream-prediction--refinery-process-equipment-failure)
5. [Project Report Summary](#5-project-report-summary)
6. [ExxonMobil Domain Knowledge & Industrial Implementation](#6-exxonmobil-domain-knowledge--industrial-implementation)
7. [Technology Stack](#7-technology-stack)
8. [Repository Structure](#8-repository-structure)

---

## 1. Project Overview

This project develops a **comprehensive machine learning-based predictive maintenance system** spanning both **upstream** (gas turbine cogeneration) and **downstream** (refinery process equipment) operations for ExxonMobil's Singapore integrated refining and petrochemical complex — the world's largest of its kind.

### Business Problem

ExxonMobil's Singapore facility operates at a capacity of **592,000 barrels per day**, powered by a cogeneration plant running **GE 9HA series gas turbines** (605 MW, 62.22% thermodynamic efficiency). The facility's electricity consumption exceeds **440 MWh**, far surpassing Singapore's mainland energy storage capacity of 200 MWh — meaning alternative power sources are unavailable during outage contingencies. A single hour of unplanned downtime costs over **$100,000**, and downstream profit fell by **$50 million** in Q2 2019 alone due to outages at three ExxonMobil refineries (Baytown, Canada, and Saudi Arabia). Unplanned CDU outages in Q2 2019 resulted in **16.4 million barrels of lost production**.

### Strategic Objective

Transition ExxonMobil's maintenance philosophy from reactive/preventive to **predictive** by deploying ML classifiers on sensor telemetry data from:

- **Upstream (Part A):** GE 9HA gas turbines in the Jurong Island cogeneration plant — predicting turbine failure before it cascades into plant-wide power loss
- **Downstream (Part B):** Critical refinery process equipment (crude distillation units, fluid catalytic cracking units, heat exchangers, compressors) — predicting equipment degradation and failure across the refining value chain

The dual-scope approach ensures end-to-end operational resilience: upstream prediction safeguards the **power supply**, while downstream prediction safeguards the **process units** that convert crude oil into high-value products.

---

## 2. ExxonMobil Operational Context

### 2.1 The Singapore Integrated Complex

ExxonMobil's Jurong Island facility is a vertically integrated refining and petrochemical complex comprising:

- **Refinery:** 592,000 bbl/day crude distillation capacity — Singapore's largest, producing fuels, lubricants, and aromatics
- **Petrochemical plants:** Ethylene crackers, polyethylene/polypropylene units, aromatics extraction, specialty chemicals
- **Cogeneration plant:** GE 9HA combined-cycle gas turbines providing captive electricity and process steam to the entire complex
- **Marine terminal:** Deep-water jetties handling VLCC tanker operations

The integration means a failure at any point — gas turbine trip, CDU shutdown, FCC malfunction — propagates across the entire value chain. A cogeneration plant trip blacks out the refinery; a CDU failure starves the FCC of feedstock; an FCC failure disrupts the petrochemical downstream.

### 2.2 Current Maintenance Framework

ExxonMobil operates under its **Operations Integrity Management System (OIMS)** — a proprietary framework established after the 1989 Exxon Valdez incident, comprising:

| Framework | Scope |
|---|---|
| **Standards of Environmental Care (SOEC)** | Environmental protection across all operations |
| **Global Pipeline Integrity (GPI)** | Pipeline corrosion monitoring and integrity management |
| **Hydrocarbon Controls Procedures (HCP)** | Leak prevention and hydrocarbon containment |
| **Safety Critical Equipment (SCE) Programme** | Classification and inspection of safety-critical assets |
| **Rotor-In Major Inspection (RIMI)** | GE gas turbine maintenance without rotor removal (saves 30-40% labour hours) |

**Current gap:** While ExxonMobil partnered with Microsoft Azure in 2019 for upstream drilling analytics and AI-driven oilfield optimisation, this cloud technology has **not yet been deployed for downstream process equipment maintenance**. The refinery and cogeneration plant still rely on scheduled (preventive) and reactive maintenance strategies.

### 2.3 Limitations of Existing Approach

| Strategy | Limitation | ExxonMobil Impact |
|---|---|---|
| **Reactive Maintenance** | Repairs only after breakdown | Unplanned shutdowns cost >$100K/hour; safety incidents (e.g., 2017 Louisiana refinery explosion — OSHA citations issued) |
| **Preventive Maintenance** | Fixed-interval inspections regardless of actual condition | Over-maintenance: unnecessary teardowns increase planned downtime; Under-maintenance: 40% of failures still occur between scheduled inspections |
| **RIMI Procedures** | Reduces but does not eliminate downtime | Can only reduce outage labour by 30-40%; does not prevent the failure event itself |

### 2.4 The Predictive Maintenance Opportunity

McKinsey & Company research indicates IoT-enabled predictive maintenance delivers:
- **30% reduction** in maintenance costs
- **30% reduction** in unplanned downtime
- **20% extension** in equipment lifespan

For ExxonMobil's Singapore complex operating at 592,000 bbl/day with crude at ~$75/bbl, even a **1% improvement in uptime** translates to approximately **$16.2 million/year** in avoided lost production.

---

## 3. Part A: Upstream Prediction — Gas Turbine Failure in Cogeneration Plants

### 3.1 Problem Statement

Predict failure of GE 9HA series gas turbines in ExxonMobil's Jurong Island cogeneration plant using operational sensor data, enabling proactive maintenance scheduling before turbine trips cascade into plant-wide power outages.

### 3.2 Dataset

**Source:** Synthetic predictive maintenance dataset modelling GE H-series gas turbine operational data (UCI ML Repository: AI4I 2020 Predictive Maintenance Dataset)

**Records:** 10,000 data points | **Features:** 10

| Feature | Description | Unit | ExxonMobil Sensor Equivalent |
|---|---|---|---|
| `UDI` | Unique identifier (1–10,000) | — | SAP Equipment ID |
| `Product ID` | Quality variant prefix (L/M/H) + serial | — | Asset tag in Maximo/SAP PM |
| `Type` | Machine quality variant: H (High), M (Medium), L (Low) | — | Turbine frame classification |
| `Air temperature [K]` | Ambient air intake temperature | Kelvin | Compressor inlet RTD (T1) |
| `Process temperature [K]` | Hot gas path / exhaust temperature | Kelvin | Turbine exhaust thermocouple (T5) |
| `Rotational speed [rpm]` | Shaft rotational speed | RPM | Shaft proximity probe / tachometer |
| `Torque [Nm]` | Mechanical torque on shaft | Nm | Strain gauge / calculated from generator load |
| `Tool wear [min]` | Cumulative operational wear time | Minutes | Equivalent Operating Hours (EOH) counter |
| `Target` | Binary failure indicator: 0 = No Failure, 1 = Failure | — | Trip/alarm event log from Mark VIe control system |
| `Failure Type` | Failure mode (Heat Dissipation, Power, Overstrain, Tool Wear, Random) | — | CMMS failure code taxonomy |

### 3.3 Class Distribution

| Class | Count | Percentage |
|---|---|---|
| No Failure (0) | 9,661 | 96.6% |
| Failure (1) | 339 | 3.4% |

**Severe class imbalance** — reflective of real-world turbine operations where failures are rare but catastrophic events.

### Machine Quality Variant Distribution

| Type | Count | Percentage | ExxonMobil Interpretation |
|---|---|---|---|
| L (Low quality) | ~6,000 | 60% | Standard-grade turbine components with baseline tolerance specs |
| M (Medium quality) | ~3,000 | 30% | Mid-grade components with enhanced coatings and alloys |
| H (High quality) | ~1,000 | 10% | Premium single-crystal blade alloys, advanced TBC coatings |

### 3.4 Data Preprocessing Pipeline

#### Step 1: Data Ingestion & Cleaning
- Loaded via `data.table::fread()` with column renaming for programmatic access
- Verified **zero missing values** across all features
- Removed non-predictive identifiers (`UDI`, `Product ID`) and `Failure Type` label (prevents target leakage)

#### Step 2: Class Imbalance Treatment — Random Undersampling
- Undersampled majority class (9,661 non-failure) to match minority class (339 failure)
- Resulting balanced dataset: **~678 records** (339 per class)
- **Rationale:** In cogeneration operations, the cost of a missed turbine trip (false negative) — cascading plant-wide blackout, emergency shutdown of CDUs/FCC, flaring, potential safety incident — vastly exceeds the cost of an unnecessary inspection (false positive). Undersampling forces models to learn genuine failure patterns rather than trivially predicting "no failure" for 96.6% accuracy.

#### Step 3: Skewness Correction
- Assessed all continuous features using `moments::skewness()`
- `Rotational_speed_rpm` exhibited **high positive skew (2.73)** — corrected via **log transformation** to 2.11
- Remaining features: within acceptable skewness bounds
- Note: Selected models without normality assumptions given residual skewness

#### Step 4: Feature Engineering
- **One-hot encoding** of `Type` into binary columns: `TypeH`, `TypeL`, `TypeM`
- **Min-Max normalisation** (0–1 scaling) on all continuous features:
  - Air_temperature_K, Process_temperature_K, Rotational_speed_rpm (log), Torque_nm, Tool_wear_min

#### Step 5: Train-Test Split
- **70/30 stratified split** via `caTools::sample.split()` on `Target`
- Random seed: 100 (reproducibility)

### 3.5 Models Evaluated

Ten model configurations were trained, evaluated, and benchmarked:

| # | Model | Category | Key Configuration |
|---|---|---|---|
| 1 | **Logistic Regression** (Baseline) | Linear | GLM, binomial family |
| 2 | **Support Vector Machine** | Cluster-based | C-classification, linear kernel |
| 3 | **CART Decision Tree** | Tree-based | Maximal tree grown (minsplit=2, cp=0), pruned via 1-SE rule |
| 4 | **Random Forest** | Ensemble (Bagging) | ntree = 30 |
| 5 | **10-Fold CV Random Forest** | Ensemble + CV | ntree = 30, k = 10 |
| 6 | **Ridge Regression** | Regularised Linear | alpha = 0, lambda optimised via CV |
| 7 | **Lasso Regression** | Regularised Linear | alpha = 1, lambda optimised via CV |
| 8 | **XGBoost** | Gradient Boosting | nrounds = 10, binary:logistic, error metric |
| 9 | **10-Fold CV XGBoost** | Gradient Boosting + CV | nrounds = 10, binary:hinge, k = 10 |
| 10 | **Neural Network** | Deep Learning | 4 hidden layers (6-6-4-2), cross-entropy loss, stepmax = 1M |

### 3.6 Evaluation Metrics

| Metric | Formula / Description | Why It Matters for Gas Turbines |
|---|---|---|
| **F1 Score** (Primary) | Harmonic mean of Precision & Recall | Balances false alarms (unnecessary shutdowns) vs. missed failures (catastrophic trips) |
| **F2 Score** | Weighted F-beta (beta=2), emphasises Recall | Penalises missed failures more heavily — critical because a turbine trip costs >$100K/hour vs. an unnecessary inspection costing ~$5K |
| **Accuracy** | (TP+TN) / Total | Overall correctness; less meaningful under class imbalance |
| **Log Loss** | -[y*log(p) + (1-y)*log(1-p)] | Measures probabilistic calibration — important for confidence-tiered alerting (yellow vs. red alarms) |

### 3.7 Results

| Method | Model | Accuracy | F1 Score | F2 Score | Log Loss |
|---|---|---|---|---|---|
| Linear | Logistic Regression (Baseline) | 0.804 | 0.810 | 0.824 | 16.955 |
| Cluster-based | Support Vector Machine | 0.799 | 0.800 | 0.802 | -34.540 |
| Regularised Linear | Ridge Regression | 0.804 | 0.802 | 0.797 | 17.320 |
| Regularised Linear | Lasso Regression | 0.804 | 0.810 | 0.824 | 17.320 |
| Tree-based | CART | 0.931 | 0.933 | 0.950 | 17.025 |
| Ensemble (Bagging) | Random Forest | 0.931 | 0.934 | 0.956 | 17.016 |
| Ensemble + CV | 10-Fold Random Forest | 0.930 | 0.934 | 0.956 | 17.016 |
| Deep Learning | Neural Network | 0.858 | 0.863 | 0.880 | **5.086** |
| **Gradient Boosting** | **XGBoost** | **0.936** | **0.938** | **0.957** | 16.609 |
| Gradient Boosting + CV | 10-Fold XGBoost | **0.971** | 0.938 | 0.957 | 16.609 |

### 3.8 Key Findings — Upstream

#### Variable Importance (Consistent Across CART, Random Forest, and XGBoost)

| Rank | Feature | CART Variable Importance | RF Mean Decrease Gini | Interpretation |
|---|---|---|---|---|
| 1 | **Rotational speed (RPM)** | ~100 (highest) | 65.0 | Primary split at normalised speed >= 0.18; low speed = compressor stall/bearing degradation |
| 2 | **Torque (Nm)** | ~95 | 67.0 | High torque + low speed = mechanical overload; inverse relationship with RPM confirmed in EDA |
| 3 | **Tool wear (min)** | ~48 | 42.0 | Cumulative degradation proxy; failure threshold at >78th percentile normalised wear |
| 4 | **Air temperature (K)** | ~18 | 25.0 | Ambient conditions affect compressor inlet density and turbine cooling effectiveness |
| 5 | **Process temperature (K)** | ~5 | 17.0 | Hot gas path temperature; less discriminative as standalone feature |
| 6-8 | **Type (H/L/M)** | Negligible | 1-3 each | Quality variant has minimal standalone predictive power |

#### CART Decision Tree — Interpretable Failure Rules

The optimal pruned CART model (after 1-SE rule pruning) reveals rules directly translatable to gas turbine alarm setpoints:

```
Rule 1: IF Rotational_speed_rpm < 0.18 (normalised) → 85% probability of failure
        [Maps to: turbine operating significantly below rated speed — compressor surge/stall zone]

Rule 2: IF Rotational_speed_rpm >= 0.18 AND Tool_wear_min >= 0.81 → failure predicted
        [Maps to: turbine at normal speed but approaching overhaul interval]

Rule 3: IF Rotational_speed_rpm < 0.18 AND Torque_nm >= 0.68 → failure predicted
        [Maps to: high mechanical load at low speed — bearing/gearbox distress]

Rule 4: IF Rotational_speed_rpm < 0.18 AND Torque_nm < 0.68 AND Air_temp >= 0.66 → investigate
        [Maps to: hot-day low-speed operation — reduced cooling margin]
```

#### Selected Model: XGBoost

XGBoost was selected as the final upstream model based on:
- **Highest F1 Score (0.938)** — best balance of precision and recall
- **Highest F2 Score (0.957)** — strongest recall performance (fewest missed failures)
- **Highest accuracy (0.971 with 10-fold CV)** — most stable generalisation
- **Computational efficiency** — gradient boosting is well-suited for real-time inference on edge devices
- **Log Loss (16.609)** — acceptable probabilistic calibration for alert tiering

**Note:** While Neural Networks achieved the best Log Loss (5.086) indicating superior probabilistic calibration, their lower F1/F2 scores and the "black box" interpretability concern led to XGBoost being preferred for operational deployment. The report recommends a **human-in-the-loop** approach where XGBoost predictions serve as a complementary decision support tool alongside engineer judgement.

### 3.9 Upstream Operational Monitoring Recommendations

Based on the variable importance findings, the following real-time monitoring priorities should be established for ExxonMobil's cogeneration gas turbines:

- **Rotational speed monitoring** on gas turbines, compressors, and pumps should be the primary trigger for predictive maintenance alerts. Speed deviations from baseline are the single most predictive failure indicator — the CART root split at normalised RPM >= 0.18 translates directly to a low-speed alarm threshold on the Mark VIe control system.
- **Torque-speed relationship monitoring** enables detection of mechanical binding, bearing wear, and compressor surge conditions before catastrophic failure. The confirmed inverse relationship between torque and RPM means that a simultaneous high-torque / low-speed condition should trigger an immediate "orange" alert.
- **Cumulative runtime tracking** supports condition-based overhaul scheduling rather than fixed-interval maintenance, reducing unnecessary teardowns on healthy equipment. The CART model's wear threshold at the 78th percentile provides an evidence-based trigger for scheduling the next RIMI inspection.

### 3.10 Upstream Deployment Recommendations

1. **Edge deployment** of the XGBoost model on existing RTU/PLC infrastructure or the GE Mark VIe turbine control system, enabling sub-second inference without cloud latency for safety-critical trip prevention
2. **Integration with OSIsoft PI historian** for real-time feature extraction from existing sensor tags — the PI-to-Azure connector enables dual-path scoring (edge for real-time, cloud for batch retraining)
3. **Alert tiering** using probability thresholds to generate confidence-graded maintenance work orders in SAP PM/Maximo:
   - P(fail) > 0.3 → Yellow: "Investigate — review sensor trends"
   - P(fail) > 0.5 → Orange: "Schedule maintenance within next planned window"
   - P(fail) > 0.8 → Red: "Immediate action — prepare for controlled shutdown"
4. **Continuous retraining pipeline** using actual failure records from the CMMS to improve model accuracy over time and prevent concept drift as turbine ageing progresses
5. **CART decision rules as field pocket cards** — the interpretable tree rules (Section 3.8) should be printed as laminated pocket reference cards for field technicians and control room operators, providing an immediate human-interpretable cross-check against the XGBoost probability output

---

## 4. Part B: Downstream Prediction — Refinery Process Equipment Failure

### 4.1 Problem Statement

Predict failure of critical refinery process equipment across ExxonMobil's Singapore downstream complex — including crude distillation units (CDUs), fluid catalytic cracking (FCC) units, hydrotreaters, reformers, heat exchangers, and rotating equipment (pumps, compressors) — using process sensor data and historical maintenance records.

While the upstream model (Part A) protects the **power supply** to the refinery, the downstream model addresses the **process units themselves**. A CDU failure at ExxonMobil's Singapore refinery impacts 592,000 bbl/day throughput; an FCC failure disrupts gasoline and petrochemical feedstock production; a heat exchanger tube leak can force an unplanned unit shutdown with environmental release risk.

### 4.2 ExxonMobil Downstream Asset Landscape

ExxonMobil's Singapore refinery operates the following critical process units, each with distinct failure modes and sensor signatures:

| Process Unit | Function | Capacity | Critical Failure Modes | Key Sensor Inputs |
|---|---|---|---|---|
| **Crude Distillation Unit (CDU)** | Primary crude oil separation into fractions | 592,000 bbl/day | Tray fouling, overhead condenser corrosion, furnace tube coking, column flooding | Column delta-P, tray temperatures, overhead reflux ratio, furnace tube metal temps |
| **Vacuum Distillation Unit (VDU)** | Heavy residue separation under vacuum | ~200,000 bbl/day | Ejector malfunction, vacuum loss, coking in transfer line | Vacuum pressure, ejector steam flow, heater outlet temperature |
| **Fluid Catalytic Cracker (FCC)** | Converts heavy gas oil to gasoline/LPG | ~150,000 bbl/day | Catalyst attrition, regenerator afterburn, cyclone erosion, slide valve failure | Riser temperature, regenerator dense bed temp, catalyst circulation rate, flue gas O2/CO |
| **Hydrocracker/Hydrotreater** | Sulfur removal, heavy oil conversion | ~100,000 bbl/day | Catalyst deactivation, reactor bed channelling, hydrogen compressor failure | Reactor delta-T across beds, H2 partial pressure, WABT, compressor vibration |
| **Catalytic Reformer** | Naphtha upgrading to high-octane reformate | ~80,000 bbl/day | Catalyst coking, chloride loss, heater tube failure | Reactor inlet/outlet temps, H2/HC ratio, chloride in recycle gas, pressure drop |
| **Heat Exchangers** (100s of units) | Heat integration across all process units | — | Tube leak, fouling, corrosion under insulation (CUI) | Shell/tube delta-T, fouling factor trend, pressure drop, vibration |
| **Centrifugal Pumps** (1000s of units) | Fluid transfer across all process units | — | Seal failure, bearing wear, impeller erosion, cavitation | Vibration (axial/radial), bearing temperature, suction pressure, discharge flow |
| **Reciprocating/Centrifugal Compressors** | Gas compression (H2, process gas, refrigerant) | — | Valve failure, piston ring wear, surge, bearing distress | Vibration, rod drop, discharge temperature, suction/discharge pressure, surge margin |

### 4.3 Downstream Dataset Design

The downstream prediction task extends the upstream methodology to refinery-specific sensor telemetry. The dataset would be sourced from ExxonMobil's **OSIsoft PI historian** (or equivalent DCS/SCADA data archive) and **SAP PM/Maximo CMMS** work order records.

#### Proposed Feature Set

| Feature Category | Features | Source System | Sampling Rate |
|---|---|---|---|
| **Process Variables** | Column temperatures (top/mid/bottom), pressure drops, flow rates, reflux ratios, heat duties | DCS (Honeywell Experion / Yokogawa CENTUM VP) | 1-second scans, aggregated to 1-minute averages |
| **Rotating Equipment** | Vibration (velocity, acceleration, displacement), bearing temperatures, seal pressures, motor current | Online condition monitoring (Bently Nevada / SKF) | Continuous (1-second) |
| **Corrosion & Integrity** | Ultrasonic wall thickness, corrosion coupon rates, ER probe readings, CUI thermal imaging | Integrity management system | Daily/weekly scans |
| **Operational Context** | Throughput (bbl/day), feed quality (API gravity, sulfur content, TAN), ambient temperature, days since last turnaround | PI historian, LIMS, planning system | Hourly / per-crude-assay |
| **Maintenance History** | Days since last overhaul, cumulative run hours, number of past failures, last work order type | SAP PM / Maximo | Event-driven |
| **Target Variable** | Binary: equipment failure within next N operating days (e.g., N=7, 14, 30) | CMMS failure notifications + operator logs | Event-driven |

#### Expected Dataset Characteristics

| Characteristic | Upstream (Gas Turbine) | Downstream (Refinery Equipment) |
|---|---|---|
| Records | 10,000 | 50,000–500,000+ (multiple units, years of historian data) |
| Features | 10 | 30–100+ (process variables per unit type) |
| Failure rate | 3.4% | 1–5% (varies by equipment class; heat exchangers ~2%, pumps ~4%, compressors ~3%) |
| Temporal structure | Independent events | **Time-series** with autocorrelation — requires windowed feature engineering |
| Multi-class potential | 5 failure types | 10–20+ failure modes per equipment class |
| Data quality challenges | Clean synthetic data | Missing values, sensor drift, tag decommissions, unit outage gaps |

### 4.4 Downstream Preprocessing Pipeline

Building on the upstream methodology, the downstream pipeline incorporates additional complexity inherent to refinery operations:

#### Step 1: Data Extraction & Integration
- Extract process historian data (PI/DCS) at 1-minute resolution for target equipment
- Join with CMMS work order data to label failure events (target variable creation)
- Join with LIMS data for crude feed quality features
- **Challenge:** Data alignment across disparate systems with different timestamps and sampling rates

#### Step 2: Temporal Feature Engineering
Unlike the upstream dataset (independent records), refinery sensor data is **time-series** — requiring:
- **Rolling window statistics:** 1hr, 8hr, 24hr, 7-day rolling mean, std, min, max for each sensor tag
- **Rate-of-change features:** First derivatives of key process variables (delta-T/hour, delta-P/hour)
- **Cumulative degradation indicators:** Fouling factor trend slope, cumulative vibration energy above baseline
- **Operational regime detection:** Startup, steady-state, turndown, shutdown classification

#### Step 3: Class Imbalance Treatment
- **SMOTE (Synthetic Minority Oversampling)** preferred over undersampling for larger downstream datasets — retains more training information
- **Cost-sensitive learning** weights in XGBoost: `scale_pos_weight` = (negative count / positive count)
- **Sliding window prediction:** Predict failure in next N days, with N tuned per equipment class based on maintenance lead time

#### Step 4: Multi-Equipment Modelling Strategy

| Approach | Description | Use Case |
|---|---|---|
| **Equipment-specific models** | Separate model per equipment class (CDU model, pump model, etc.) | When failure signatures differ fundamentally between equipment types |
| **Federated/Transfer learning** | Pre-train on one unit, fine-tune on similar units | Leverage data from multiple similar compressors or pumps across the complex |
| **Hierarchical model** | Unit-level model feeds into plant-level availability prediction | Aggregate risk scoring for turnaround planning |

### 4.5 Downstream Models

The same model families evaluated in the upstream analysis would be applied, with extensions for time-series and larger datasets:

| Model | Upstream Config | Downstream Extension |
|---|---|---|
| **Logistic Regression** | Standard GLM | Regularised (elastic net) with temporal features |
| **CART / Random Forest** | ntree=30, Gini splits | Larger forests (ntree=500+), deeper trees, feature selection via Boruta |
| **XGBoost** | nrounds=10 | nrounds=500+, Bayesian hyperparameter tuning, early stopping |
| **Neural Network** | 4-layer MLP (6-6-4-2) | **LSTM / CNN-LSTM hybrid** for temporal pattern recognition |
| **Additional: Isolation Forest** | N/A | Unsupervised anomaly detection for novel failure modes |
| **Additional: Survival Analysis** | N/A | Time-to-failure prediction (Cox PH / DeepSurv) for remaining useful life estimation |

### 4.6 Expected Downstream Results & Failure Signatures

Based on ExxonMobil domain knowledge and published refinery predictive maintenance benchmarks:

#### CDU / Distillation Columns
- **Primary failure predictor:** Column differential pressure trend (fouling progression follows a characteristic exponential curve)
- **Secondary:** Overhead condenser duty decline, tray temperature profile distortion
- **Expected model accuracy:** 85–92% (fouling is gradual and highly predictable)
- **Lead time achievable:** 2–4 weeks before throughput impact

#### FCC Unit
- **Primary failure predictor:** Regenerator dense bed temperature excursion + catalyst-to-oil ratio deviation
- **Secondary:** Cyclone pressure drop increase (erosion), slide valve differential pressure instability
- **Expected model accuracy:** 80–88% (more complex multi-mode failure landscape)
- **Lead time achievable:** 3–7 days for catalyst issues; hours for mechanical failures

#### Heat Exchangers
- **Primary failure predictor:** Fouling factor trend (UA degradation rate), approach temperature increase
- **Secondary:** Shell-side pressure drop increase, tube-side outlet temperature deviation from design
- **Expected model accuracy:** 88–94% (fouling is the dominant, highly predictable failure mode)
- **Lead time achievable:** 1–6 months (fouling is slow; cleaning can be scheduled during planned outages)

#### Rotating Equipment (Pumps & Compressors)
- **Primary failure predictor:** Vibration signature change (amplitude, frequency spectrum shift) — directly analogous to upstream torque/RPM findings
- **Secondary:** Bearing temperature trend, seal oil delta-P, motor current draw
- **Expected model accuracy:** 90–95% (vibration-based diagnostics is the most mature PDM discipline)
- **Lead time achievable:** 1–4 weeks for bearing wear; hours for seal failures

### 4.7 Downstream-Specific Considerations

#### Turnaround (TAR) Planning Integration
ExxonMobil's Singapore refinery undergoes **major turnarounds every 4–6 years** per unit, with TARs typically lasting 30–45 days and costing $50–200M per event. Downstream predictive maintenance models feed directly into TAR scope optimisation:
- Equipment predicted to survive until the next TAR can be **deferred** (avoiding unnecessary mid-cycle shutdowns)
- Equipment showing accelerating degradation can be **pulled into the TAR scope** (avoiding unplanned outage before the next TAR)
- Remaining Useful Life (RUL) estimates from survival models directly inform the **TAR scheduling window**

#### Crude Slate Sensitivity
Unlike upstream turbines running on consistent natural gas, downstream equipment condition is heavily influenced by **crude oil quality changes**:
- High-TAN (Total Acid Number) crudes accelerate naphthenic acid corrosion in CDU overheads
- High-sulfur crudes increase hydrotreater catalyst deactivation rates
- Opportunity crudes with high asphaltene content accelerate heat exchanger fouling
- **Implication:** Crude assay data must be included as a model feature; models must handle **concept drift** as crude slates change

#### Safety & Environmental Regulatory Context
ExxonMobil operates under Singapore's **Workplace Safety and Health Act** and **Environmental Protection and Management Act**. Equipment failures in downstream units carry regulatory consequences:
- Flaring during emergency shutdowns triggers **NEA (National Environment Agency)** reporting requirements
- Hydrocarbon releases require **MOM (Ministry of Manpower)** incident reporting
- Predictive maintenance reduces both safety incidents and regulatory exposure — the report notes that **30%+ of major O&G accidents are triggered by inadequate maintenance** (Nwankwo et al., 2021)

---

## 5. Project Report Summary

### 5.1 Executive Summary (from report)

The report proposes integrating ML-based predictive maintenance with ExxonMobil's existing OIMS maintenance framework to reduce unplanned downtime in the Singapore cogeneration and downstream complex. **XGBoost was selected as the final model** (highest accuracy: 0.971, highest F1: 0.938, highest F2: 0.957) for deployment via a **Microsoft Azure cloud architecture** with a **human-in-the-loop** decision framework.

### 5.2 Key Report Findings

#### Model Performance Hierarchy
1. **Non-linear models decisively outperform linear models** — CART (0.933 F1) through XGBoost (0.938 F1) vs. Logistic Regression (0.810 F1), confirming that turbine failure is governed by non-linear interactions between operating variables
2. **Ensemble methods provide the best accuracy-stability trade-off** — Random Forest and XGBoost both exceed 0.93 F1; 10-fold CV XGBoost achieves 0.971 accuracy
3. **Neural Networks excel at probabilistic calibration** (Log Loss: 5.086 vs. 16+ for all other models) but underperform on F1/F2, likely due to insufficient training data (678 balanced records) for the 4-layer architecture
4. **Linear models (Logistic, SVM, Ridge, Lasso) plateau at ~0.80 F1** — confirming the failure decision boundary is fundamentally non-linear

#### Variable Importance Consensus
All tree-based and ensemble models agree:
- **Rotational speed and torque are the dominant predictors** (combined importance >60% across all models)
- Their **inverse relationship** (confirmed in EDA) is the key failure signature: high torque at low speed = mechanical distress
- **Tool wear** (cumulative operating hours) acts as a degradation accumulator
- **Temperature features** are secondary contributors
- **Machine type** adds minimal predictive power as a standalone feature

#### Interpretability vs. Accuracy Trade-off
The report explicitly addresses the **"black box" problem** for industrial deployment:
- **CART** provides human-readable rules directly translatable to alarm setpoints — recommended for **field engineer decision support**
- **XGBoost** provides highest accuracy but requires trust-building via the proposed **human-in-the-loop framework** — recommended for **control room automated alerting**
- **42% of analytics results go unused** by business decision-makers (SAS survey cited in report) — the human-in-the-loop approach is essential for adoption

### 5.3 Proposed Implementation Architecture (from report)

```
[Gas Turbine Sensors] → [Azure IoT Hub] → [Azure Data Explorer] → [XGBoost Model] → [Azure Monitor Alerts]
        |                                          |                        |                    |
        |                                    [PI Historian]           [Probability Score]   [Control Room]
        |                                          |                        |                    |
        └── Torque, RPM, Temp,              [Dashboard/PBI]          [Human-in-the-Loop]  [Work Order in
            Wear sensors                                              Engineer Review       SAP PM/Maximo]
```

**Two-factor authentication for maintenance decisions:**
1. **Factor 1:** XGBoost model probability score (>50% = alert, confidence-tiered)
2. **Factor 2:** Engineer's domain expertise validation — reviewing specific sensor trends (especially torque-speed relationship) before confirming maintenance scheduling

### 5.4 Report Limitations Acknowledged

| Limitation | Impact | Proposed Mitigation |
|---|---|---|
| **Missing dynamic features** | Dataset lacks turbine age, operational hours, maintenance history, seasonal variables | Additional data collection from CMMS and historian; include seasonality features |
| **Seasonal variation** | Model trained on potentially single-season data; summer vs. winter operating conditions differ significantly | Collect multi-year seasonal data; retrain models periodically to prevent concept drift |
| **Spurious correlations** | Variable importance shows correlation, not causation; confounders may exist | Pilot study with domain experts; validate findings against mechanical engineering first principles |
| **Synthetic dataset** | Data is generated, not from actual ExxonMobil turbines | Retrain on real operational data before production deployment |

### 5.5 Alternative Advanced Methods Proposed

| Method | Application | Advantage over Current Models |
|---|---|---|
| **CNN-LSTM Hybrid** | Time-series classification of sequential sensor data | Captures temporal dependencies (current models treat records as independent) |
| **Deep Reinforcement Learning (DRL)** | Optimise dynamic maintenance scheduling | Learns optimal maintenance actions via Markov Decision Process; minimises total cost over time |
| **Ensemble of Ensembles** | Combine XGBoost + LSTM + domain rules | Leverages strengths of multiple paradigms |

### 5.6 Value Proposition (from report)

| Stakeholder | Value |
|---|---|
| **ExxonMobil Management** | Reduced unplanned downtime, optimised maintenance spend, increased asset utilisation, strategic resource allocation |
| **Workers** | Reduced workplace accident risk from turbine malfunctions in high-energy cogeneration facilities |
| **Engineers** | Variable importance insights for targeted sensor monitoring; hidden pattern discovery augments domain expertise |
| **Singapore Government** | Economic stability (25% of manufacturing output from energy/chemicals), environmental protection (reduced flaring/releases), geopolitical energy security (Singapore as Asia's oil trading hub) |

---

## 6. ExxonMobil Domain Knowledge & Industrial Implementation

### 6.1 ExxonMobil's Digital Transformation Journey

ExxonMobil's adoption of predictive analytics fits within a broader digital transformation roadmap:

| Year | Milestone | Scope |
|---|---|---|
| 2019 | **Microsoft Azure partnership** | Cloud computing for upstream drilling optimisation, AI-driven completions, leak detection |
| 2020 | **Digital twin programme** | Real-time simulation of refinery process units for "what-if" scenario testing |
| 2021 | **Advanced process control (APC) expansion** | Model predictive control (MPC) on CDUs, FCCs — optimises yield/energy in real-time |
| 2022 | **OIMS digital integration** | Digitisation of integrity management records, mobile inspection workflows |
| 2023 | **Enterprise data lake (Azure Synapse)** | Centralised data platform aggregating PI historian, LIMS, SAP, and drone/satellite imagery |
| **Proposed** | **Predictive maintenance deployment** | ML-based failure prediction integrated with Azure IoT + OIMS framework |

### 6.2 Operational Integrity Management System (OIMS) Integration

The predictive maintenance system must integrate within ExxonMobil's 11 OIMS elements:

| OIMS Element | Predictive Maintenance Integration Point |
|---|---|
| **1. Management Leadership** | Executive dashboard showing plant-wide equipment health scores and predicted failure risk |
| **2. Risk Assessment** | ML-predicted failure probabilities feed into quantitative risk assessments for Process Hazard Analysis (PHA) |
| **3. Facilities Design** | Historical failure pattern data informs design improvements during turnarounds and capital projects |
| **4. Information/Documentation** | Model predictions, engineer overrides, and outcomes logged in CMMS for continuous improvement |
| **5. Personnel & Training** | Control room operators and maintenance engineers trained on interpreting model outputs and CART decision rules |
| **6. Operations & Maintenance** | **Core integration point:** Automated work order generation in SAP PM based on predicted failure probability thresholds |
| **7. Management of Change (MOC)** | Model retraining, threshold changes, and new sensor additions governed under MOC procedures |
| **8. Third-Party Services** | Model predictions shared with GE FieldCore for coordinating RIMI inspections and parts procurement |
| **9. Incident Investigation** | Post-failure analysis compares model predictions against actual failure events for continuous improvement |
| **10. Community Awareness** | Reduced flaring and emergency venting from prevented unplanned shutdowns improves community relations |
| **11. Operations Integrity Assessment** | Annual model performance audit (accuracy drift, false negative rate trending) as part of OIMS self-assessment |

### 6.3 ExxonMobil-Specific Technical Architecture

#### Sensor Infrastructure (Already Deployed)

ExxonMobil's Singapore complex already has comprehensive instrumentation:

| System | Vendor | Data Points | Sampling |
|---|---|---|---|
| **DCS (Distributed Control System)** | Honeywell Experion / Yokogawa | ~50,000 process I/O tags | 1-second scan |
| **SIS (Safety Instrumented System)** | Triconex | ~5,000 safety-critical I/O | Continuous |
| **Vibration monitoring** | Bently Nevada (GE) / SKF | ~2,000 rotating equipment channels | Continuous |
| **Gas turbine control** | GE Mark VIe | ~500 per turbine | 100ms scan |
| **Corrosion monitoring** | Emerson/Honeywell ER probes | ~200 corrosion loops | Hourly |
| **Data historian** | OSIsoft PI System | Aggregates all above | Configurable (typically 1-min average) |

**Key insight:** The data infrastructure for predictive maintenance already exists — the missing component is the **analytics layer** (ML models + alerting logic) that this project provides.

#### Proposed Production Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        ExxonMobil Singapore Complex                         │
│                                                                             │
│  [GE 9HA Turbines]  [CDU/FCC/HCU]  [Pumps/Compressors]  [Heat Exchangers] │
│         │                  │                │                    │           │
│    [Mark VIe]         [DCS Tags]     [Bently Nevada]      [ER Probes]      │
│         │                  │                │                    │           │
│         └──────────────────┴────────────────┴────────────────────┘          │
│                                    │                                        │
│                          [OSIsoft PI Historian]                              │
│                                    │                                        │
│              ┌─────────────────────┼────────────────────┐                   │
│              │                     │                    │                    │
│        [PI-to-Azure          [SAP PM/Maximo]     [LIMS Crude               │
│         Connector]            Work Orders         Assay Data]              │
└──────────────┼─────────────────────┼────────────────────┼───────────────────┘
               │                     │                    │
               ▼                     ▼                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                          Microsoft Azure Cloud                               │
│                                                                              │
│  [Azure IoT Hub] → [Azure Data Explorer] → [Azure Synapse Analytics]        │
│                              │                        │                      │
│                     [Feature Engineering]    [Model Training Pipeline]        │
│                     (rolling stats, RoC,     (Azure ML: XGBoost, LSTM,       │
│                      regime detection)        Random Forest)                 │
│                              │                        │                      │
│                     [Real-time Scoring] ←─── [Model Registry]                │
│                              │                                               │
│                     [Azure Monitor Alerts]                                    │
│                      ├── Yellow: P(fail) > 0.3 → "Investigate"              │
│                      ├── Orange: P(fail) > 0.5 → "Schedule Maintenance"     │
│                      └── Red:    P(fail) > 0.8 → "Immediate Action"         │
│                              │                                               │
│                     [Power BI Dashboard]                                      │
│                     (Equipment health scores,                                │
│                      trend charts, failure                                    │
│                      probability heatmaps)                                   │
└──────────────────────┬───────────────────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                    ExxonMobil Control Room / Maintenance Dept                 │
│                                                                              │
│  [Human-in-the-Loop Review] → [SAP PM Work Order Creation] → [Field Crew]   │
│                                                                              │
│  Two-Factor Decision:                                                        │
│  1. XGBoost probability + CART interpretable rules                           │
│  2. Engineer domain validation (sensor trend review)                         │
│                                                                              │
│  Feedback loop: Actual outcomes logged → Model retraining trigger            │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 6.4 Cost-Benefit Analysis

#### Conservative Estimate for ExxonMobil Singapore

| Parameter | Value | Source |
|---|---|---|
| Refinery capacity | 592,000 bbl/day | ExxonMobil public filings |
| Gross refining margin (Singapore) | ~$8–12/bbl | Platts benchmark |
| Revenue at risk per day of downtime | $4.7M–$7.1M | Capacity x margin |
| Average unplanned downtime events/year | 3–5 (industry average) | Wood Mackenzie |
| Average duration per event | 3–7 days | ExxonMobil 2019 Q2 earnings report |
| **Annual cost of unplanned downtime** | **$42M–$250M** | Range estimate |
| Predictive maintenance cost reduction | 30% (McKinsey) | Industry benchmark |
| **Estimated annual savings** | **$12.6M–$75M** | Conservative to optimistic |
| Implementation cost (Year 1) | $3M–$8M | Azure infrastructure + model development + integration |
| **ROI (Year 1)** | **3:1 to 9:1** | Conservative estimate |

### 6.5 Failure Mode Mapping to Model Features

#### Upstream (Gas Turbine) — Failure Physics

| Failure Type in Dataset | Physical Mechanism | Key Sensor Signature | ExxonMobil Action |
|---|---|---|---|
| **Heat Dissipation Failure** | Insufficient cooling; hot gas path over-temperature | Process temp >> Air temp spread widens | Check cooling air passages, inspect TBC coating |
| **Power Failure** | Loss of electrical generation capacity | RPM drop + torque fluctuation | Generator inspection, exciter system check |
| **Overstrain Failure** | Mechanical overload beyond design limits | High torque + low RPM simultaneously | Reduce load, inspect bearings/gearbox, check alignment |
| **Tool Wear Failure** | Cumulative component degradation | Tool wear exceeds threshold + other variables deteriorating | Schedule RIMI, replace hot-section components |
| **Random Failure** | Stochastic events (FOD, manufacturing defects) | No clear sensor precursor | Hardest to predict; ensemble models capture weak signals |

#### Downstream (Refinery Equipment) — Failure Physics

| Equipment | Dominant Failure Mode | Physical Mechanism | Predictive Signal |
|---|---|---|---|
| **CDU** | Overhead corrosion | HCl condensation in overhead system from chloride in crude | Overhead pH trending down, iron counts in overhead water rising |
| **CDU** | Furnace tube coking | Thermal cracking deposits on tube ID | Tube metal temperatures rising, pass flow imbalance increasing |
| **FCC** | Catalyst deactivation | Heavy metals (Ni, V) poisoning + hydrothermal deactivation | Conversion dropping at constant severity, catalyst addition rate increasing |
| **FCC** | Regenerator afterburn | Incomplete coke burn in dense bed, CO burning in dilute phase | Dilute phase temp spike, CO/CO2 ratio shift |
| **Heat Exchanger** | Fouling | Deposition of asphaltenes, salts, corrosion products | UA declining, approach temperature increasing on predictable curve |
| **Pump** | Bearing wear | Lubrication degradation, misalignment, imbalance | Vibration amplitude increase at 1x, 2x running speed; bearing temp trend |
| **Compressor** | Surge | Operating point moves left of surge line | Discharge pressure oscillation, anti-surge valve opening frequency |

---

## 7. Technology Stack

| Component | Upstream (Implemented) | Downstream (Proposed Extension) |
|---|---|---|
| **Language** | R (v4.x) | R + Python (scikit-learn, TensorFlow/PyTorch for LSTM) |
| **Data Handling** | `data.table` | `data.table` + `pandas` + Apache Spark (for historian-scale data) |
| **Visualisation** | `ggplot2`, `rpart.plot` | `ggplot2` + `plotly` + Power BI dashboards |
| **Statistical Analysis** | `moments` (skewness) | `moments` + `tseries` (stationarity tests) + `forecast` (time-series decomposition) |
| **ML Models** | `glm`, `e1071`, `rpart`, `randomForest`, `glmnet`, `xgboost`, `neuralnet` | Same + `keras`/`torch` (LSTM/CNN), `survival` (Cox PH), `isotree` (Isolation Forest) |
| **Model Evaluation** | `MLmetrics`, `caret` | `MLmetrics`, `caret` + `PRROC` (PR curves), `survminer` (survival curves) |
| **Data Splitting** | `caTools` | `caTools` + time-aware train/test splitting (no future leakage) |
| **BI Dashboard** | Power BI (`BC2407 PBI2.pbix`) | Power BI + Azure Monitor dashboards |
| **Cloud Infrastructure** | Local R environment | Azure IoT Hub + Azure Data Explorer + Azure ML |
| **Data Historian** | CSV flat file | OSIsoft PI → Azure Data Explorer connector |
| **CMMS Integration** | N/A | SAP PM / Maximo API for automated work order generation |

---

## 8. Repository Structure

```
Predictive_Maintenance_Upstream_Downstream/
│
├── BC2407 Project (for group) v29-2-2024 upstream.R    # Part A: Upstream gas turbine ML pipeline
│                                                         # (Logistic Reg, SVM, CART, RF, Ridge/Lasso,
│                                                         #  XGBoost, Neural Network)
│
├── predictive_maintenance.csv                            # Source dataset: 10,000 GE 9HA turbine records
│
├── BC2407 Project.docx                                   # Full project report (11 sections + appendices)
│                                                         # Executive summary, methodology, model results,
│                                                         # industrial implementation, limitations,
│                                                         # alternative methods, value proposition
│
├── BC2407 Dataset Description.docx                       # Dataset documentation & data dictionary
│                                                         # GE 9HA series turbine context
│
├── BC2407 PBI2.pbix                                      # Power BI dashboard
│                                                         # Variable distributions, summary statistics,
│                                                         # per-variable deep-dive pages
│
├── Diagrams/
│   ├── Frequency Barplot for Power Turbine Failure.png   # Target class distribution (96.6% vs 3.4%)
│   ├── Frequency Barplot of Power Turbine Quality Variant.png  # Machine type: L(60%), M(30%), H(10%)
│   ├── Optimal CART Model.png                            # Pruned CART tree (RPM >= 0.18 as root split)
│   ├── Variable Importance Bar Chart.png                 # CART: RPM > Torque > Wear > AirTemp > ProcTemp
│   ├── Variable Importance Plot.png                      # RF MeanDecreaseGini: Torque ≈ RPM >> Wear
│   └── Neural Net.pdf                                    # Neural network architecture (8→6→6→4→2→1)
│
└── PROJECT_DESCRIPTION.md                                # This file
```

---

*This project was developed as part of NTU BC2407 Analytics II (AY2023/24 Semester 2), framed within the operational context of ExxonMobil's Singapore integrated refining and petrochemical complex. The upstream prediction model (Part A) is fully implemented in R. The downstream prediction framework (Part B) represents the natural extension to refinery process equipment, designed based on ExxonMobil operational domain knowledge and industry best practices for refinery predictive maintenance.*
