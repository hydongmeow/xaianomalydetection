"""
piloss.py
=========
Physics rule catalogue, residual calculator, and the composite
physics-informed loss used by the ablation study.

Components
----------
    PhysicsRuleSet            per-plant rule catalogue (SWaT / WADI)
    PhysicsResidualCalculator differentiable residual per rule
    TemporalConsistencyLoss   rate-of-change penalty
    CompositeLoss             recon + physics + temporal + graph
    KendallGal / GradNorm / DWA   multi-task weighting
"""

import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# ──────────────────────────────────────────────────────────────────────────────
# PHYSICS RULE CATALOG
# ──────────────────────────────────────────────────────────────────────────────

class RuleType(Enum):
    EQUALITY = "eq"      # x == target → anomaly
    GREATER_THAN = "gt"  # x > threshold → anomaly
    LESS_THAN = "lt"     # x < threshold → anomaly
    MASS_BALANCE = "mass"  # tank continuity
    VALVE_FLOW = "valve"   # closed valve → zero flow
    PUMP_FLOW = "pump"     # pump state ↔ flow coupling
    TEMPORAL_RATE = "temporal"  # level/pressure smoothness

@dataclass
class PhysicsRule:
    rule_type: RuleType
    features: List[str]
    params: Dict = field(default_factory=dict)
    weight: float = 1.0
    description: str = ""

    def __post_init__(self):
        if self.params is None:
            self.params = {}

class PhysicsRuleSet:
    """
    Catalog of physics rules for water treatment plants (SWaT/WADI).
    Supports both hardcoded anomaly rules and physics-law residuals.
    """

    def __init__(self, feature_names: List[str], plant: str = "swat"):
        """
        Parameters
        ----------
        feature_names : list of str
            All sensor/actuator names in order.
        plant : str
            'swat' or 'wadi' — affects which rules are instantiated.
        """
        self.feature_names = feature_names
        self.plant = plant.lower()
        self.name_to_idx = {name: i for i, name in enumerate(feature_names)}
        self.rules: List[PhysicsRule] = []
        self._build_rules()

    def _build_rules(self):
        """Populate rule set based on plant type."""
        if self.plant == "swat":
            self._build_swat_rules_scaled()
        elif self.plant == "wadi":
            self._build_wadi_rules()
        else:
            warnings.warn(f"Unknown plant {self.plant}; no rules loaded.")

    def print_rules(self) -> None:
        """Print a compact summary of all loaded rules with their thresholds."""
        print(f"\nPhysicsRuleSet  plant={self.plant!r}  rules={len(self.rules)}")
        print(f"{'#':<4}  {'type':<14}  {'features':<30}  {'params'}")
        print("-" * 75)
        for i, r in enumerate(self.rules):
            feat_str  = ", ".join(r.features)
            param_str = "  ".join(f"{k}={v}" for k, v in r.params.items())
            print(f"{i:<4}  {r.rule_type.value:<14}  {feat_str:<30}  {param_str}"
                  f"  w={r.weight}")
        print()

    def _build_swat_rules(self):
        """
        ========================
        After MinMax or StandardScaler normalisation the raw engineering-unit
        thresholds used in mass-balance, valve-flow, pump-flow, and temporal-
        rate rules (tank_area, flow_threshold, max_rate …) become meaningless.
        """

        """
        SWaT physics rules derived from the empirical distribution of SCALED
        (MinMax-normalised) sensor readings on clean training data.
        """

        # ══════════════════════════════════════════════════════════════════════
        # GROUP 1 — Flow reduction anomalies
        # ══════════════════════════════════════════════════════════════════════

        # FIT501 — RO permeate flow
        if "FIT501" in self.name_to_idx:
            self.rules.append(PhysicsRule(
                rule_type=RuleType.LESS_THAN,
                features=["FIT501"],
                params={"threshold": 0.0},
                weight=1.2,
                description=(
                    "FIT501 < 0.00077: RO permeate flow drop → attack. "
                )
            ))

        # FIT502 — RO concentrate/waste flow
        if "FIT502" in self.name_to_idx:
            self.rules.append(PhysicsRule(
                rule_type=RuleType.LESS_THAN,
                features=["FIT502"],
                params={"threshold": 0.0},
                weight=1.2,
                description=(
                    "FIT502 < 0.00064: RO concentrate flow drop → attack. "
                )
            ))

        # FIT503 — UF permeate flow
        if "FIT503" in self.name_to_idx:
            self.rules.append(PhysicsRule(
                rule_type=RuleType.LESS_THAN,
                features=["FIT503"],
                params={"threshold": 0},
                weight=1.2,
                description=(
                    "FIT503 < 0.001: UF permeate flow drop → attack. "
                )
            ))
        # ══════════════════════════════════════════════════════════════════════
        # GROUP 2 — Pressure drop anomalies
        # ══════════════════════════════════════════════════════════════════════

        # PIT501 — RO inlet pressure
        if "PIT501" in self.name_to_idx:
            self.rules.append(PhysicsRule(
                rule_type=RuleType.LESS_THAN,
                features=["PIT501"],
                params={"threshold": 9.468},
                weight=1.2,
                description=(
                    "PIT501 < 9.468: RO inlet pressure drop → attack. "
                )
            ))

        # PIT503 — RO permeate pressure
        if "PIT503" in self.name_to_idx:
            self.rules.append(PhysicsRule(
                rule_type=RuleType.LESS_THAN,
                features=["PIT503"],
                params={"threshold": 3.14},
                weight=1.2,
                description=(
                    "PIT503 < 3.14: RO permeate pressure drop → attack. "
                )
            ))

        # ══════════════════════════════════════════════════════════════════════
        # GROUP 3 — Level drop anomaly  (LESS_THAN at p0.001 of normal)
        # LIT401 is the RO feed tank.  Normal operation keeps it at ≈ 86%
        # of range.  When upstream pumps are attacked the level drops to ≈ 42%.
        # ══════════════════════════════════════════════════════════════════════

        # LIT401 — RO feed tank level
        if "LIT401" in self.name_to_idx:
            self.rules.append(PhysicsRule(
                rule_type=RuleType.LESS_THAN,
                features=["LIT401"],
                params={"threshold": 240},
                weight=0.8,   # lower weight — marginal separation
                description=(
                    "LIT401 < 240: RO feed tank critically low → attack. "
                )
            ))

        # ══════════════════════════════════════════════════════════════════════
        # GROUP 4 — Chemical spike anomalies
        # ══════════════════════════════════════════════════════════════════════

        # AIT402 — RO conductivity spike
        if "AIT402" in self.name_to_idx:
            self.rules.append(PhysicsRule(
                rule_type=RuleType.GREATER_THAN,
                features=["AIT402"],
                params={"threshold": 0.0},
                weight=1.5,   # highest weight — strongest discriminator
                description=(
                    "AIT402 > 328: RO conductivity spike → attack. "
                )
            ))

        # ══════════════════════════════════════════════════════════════════════
        # GROUP 5 — Actuator state anomalies
        # ══════════════════════════════════════════════════════════════════════

        # P204 == P206 → fault  (same as _build_swat_rules, scale-invariant)
        # if all(f in self.name_to_idx for f in ["P204", "P206"]):
        #     self.rules.append(PhysicsRule(
        #         rule_type=RuleType.EQUALITY,
        #         features=["P204", "P206"],
        #         params={"tolerance": 0.5},
        #         weight=1.0,
        #         description=(
        #             "P204 == P206: complementary pump control signals in same state → fault. "
        #             "n_std(P206)=0.0 in normal; any deviation is anomalous."
        #         )
        #     ))

    def _build_swat_rules_scaled(self):
        """
        SWaT physics rules for MinMax-scaled data [0, 1].

        All thresholds are expressed in scaled space where
        x_scaled = (x_raw - n_min) / (n_max - n_min).
        The scaler MUST be fit on clean normal training data only.

        Threshold derivation (from swat_analysis.csv):
          LESS_THAN  → p0.001 of the NORMAL distribution in scaled space,
                       i.e. n_mean_sc - 3.09 * n_std_sc.
                       This sits between normal (high) and attack (low) values
                       and ONLY fires when the sensor drops significantly below
                       its normal operating range.
                       MUST be > 0 — threshold=0.0 produces clamp(0-x)=0 for
                       all x≥0, making the rule a permanent no-op.

          GREATER_THAN → p0.999 of the NORMAL distribution in scaled space,
                         i.e. n_mean_sc + 3.09 * n_std_sc.
                         MUST be > 0 — threshold=0.0 produces clamp(x-0)=x,
                         which fires on EVERY sample (including normal) because
                         all scaled values are positive, inflating scores uniformly
                         and collapsing precision to the attack base rate (~29%).

          EQUALITY → tolerance=0.5 (scale-invariant for binary {0,1} sensors).

        Separation values confirm each rule has attack_mean clearly on the
        anomalous side of the threshold (separation > 0.10 in scaled units).
        """

        # ══════════════════════════════════════════════════════════════════════
        # GROUP 1 — Flow reduction anomalies  (LESS_THAN at p0.001 of normal)
        #
        # All RO/UF flow sensors have n_mean_sc ≈ 0.94–0.98, n_std_sc ≈ 0.06.
        # Attack mean_sc ≈ 0.38–0.40.  p0.001 ≈ mean - 3.09*std ≈ 0.75–0.79.
        # Threshold sits between normal (0.94+) and attack (0.38–0.40).
        # Residual = clamp(threshold - x, min=0):
        #   x > threshold (normal range) → 0  ✓ no false alarm
        #   x < threshold (attack range) → positive, proportional to drop  ✓
        # ══════════════════════════════════════════════════════════════════════

        # FIT501 — RO permeate flow
        # n_mean_sc=0.977  n_std_sc=0.060  p0.001_sc=0.792  a_mean_sc=0.399
        if "FIT501" in self.name_to_idx:
            self.rules.append(PhysicsRule(
                rule_type=RuleType.LESS_THAN,
                features=["FIT501"],
                params={"threshold": 0.79},
                weight=1.2,
                description=(
                    "FIT501 < 0.79 (scaled p0.001 of normal): "
                    "RO permeate flow drop → attack. "
                    "Normal mean_sc=0.977, attack mean_sc=0.399. Sep=0.39."
                )
            ))

        # FIT502 — RO concentrate/waste flow
        # n_mean_sc=0.938  n_std_sc=0.059  p0.001_sc=0.756  a_mean_sc=0.384
        if "FIT502" in self.name_to_idx:
            self.rules.append(PhysicsRule(
                rule_type=RuleType.LESS_THAN,
                features=["FIT502"],
                params={"threshold": 0.75},
                weight=1.2,
                description=(
                    "FIT502 < 0.75 (scaled p0.001 of normal): "
                    "RO concentrate flow drop → attack. "
                    "Normal mean_sc=0.938, attack mean_sc=0.384. Sep=0.37."
                )
            ))

        # FIT503 — UF permeate flow
        # n_mean_sc=0.960  n_std_sc=0.059  p0.001_sc=0.777  a_mean_sc=0.382
        if "FIT503" in self.name_to_idx:
            self.rules.append(PhysicsRule(
                rule_type=RuleType.LESS_THAN,
                features=["FIT503"],
                params={"threshold": 0.77},
                weight=1.2,
                description=(
                    "FIT503 < 0.77 (scaled p0.001 of normal): "
                    "UF permeate flow drop → attack. "
                    "Normal mean_sc=0.960, attack mean_sc=0.382. Sep=0.39."
                )
            ))

        # ══════════════════════════════════════════════════════════════════════
        # GROUP 2 — Pressure drop anomalies  (LESS_THAN at p0.001 of normal)
        #
        # PIT501/503: n_mean_sc≈0.944, attack mean_sc≈0.383.
        # p0.001≈0.76 sits between them.
        # ══════════════════════════════════════════════════════════════════════

        # PIT501 — RO inlet pressure
        # n_mean_sc=0.944  n_std_sc=0.056  p0.001_sc=0.771  a_mean_sc=0.383
        if "PIT501" in self.name_to_idx:
            self.rules.append(PhysicsRule(
                rule_type=RuleType.LESS_THAN,
                features=["PIT501"],
                params={"threshold": 0.76},
                weight=1.2,
                description=(
                    "PIT501 < 0.76 (scaled p0.001 of normal): "
                    "RO inlet pressure drop → attack. "
                    "Normal mean_sc=0.944, attack mean_sc=0.383. Sep=0.38."
                )
            ))

        # PIT503 — RO permeate pressure
        # n_mean_sc=0.942  n_std_sc=0.058  p0.001_sc=0.763  a_mean_sc=0.382
        if "PIT503" in self.name_to_idx:
            self.rules.append(PhysicsRule(
                rule_type=RuleType.LESS_THAN,
                features=["PIT503"],
                params={"threshold": 0.76},
                weight=1.2,
                description=(
                    "PIT503 < 0.76 (scaled p0.001 of normal): "
                    "RO permeate pressure drop → attack. "
                    "Normal mean_sc=0.942, attack mean_sc=0.382. Sep=0.38."
                )
            ))

        # ══════════════════════════════════════════════════════════════════════
        # GROUP 3 — Level drop anomaly  (LESS_THAN at p0.001 of normal)
        #
        # LIT401: n_mean_sc=0.858, attack mean_sc=0.421. p0.001≈0.527.
        # Marginal separation (0.10) but included — clear physical meaning.
        # ══════════════════════════════════════════════════════════════════════

        # LIT401 — RO feed tank level
        # n_mean_sc=0.858  n_std_sc=0.107  p0.001_sc=0.527  a_mean_sc=0.421
        if "LIT401" in self.name_to_idx:
            self.rules.append(PhysicsRule(
                rule_type=RuleType.LESS_THAN,
                features=["LIT401"],
                params={"threshold": 0.52},
                weight=0.8,
                description=(
                    "LIT401 < 0.52 (scaled p0.001 of normal): "
                    "RO feed tank critically low → attack. "
                    "Normal mean_sc=0.858, attack mean_sc=0.421. Sep=0.10."
                )
            ))

        # ══════════════════════════════════════════════════════════════════════
        # GROUP 4 — Chemical spike  (GREATER_THAN at p0.999 of normal)
        #
        # AIT402: n_mean_sc=0.110, n_std_sc=0.080.
        # p0.999 = 0.110 + 3.09*0.080 = 0.357 ≈ 0.35.
        # Attack mean_sc=0.587 >> 0.35 → strong separation (0.23 on mean,
        # 0.68 on peak).
        # ══════════════════════════════════════════════════════════════════════

        # AIT402 — RO conductivity spike
        # n_mean_sc=0.110  n_std_sc=0.080  p0.999_sc=0.357  a_mean_sc=0.587
        if "AIT402" in self.name_to_idx:
            self.rules.append(PhysicsRule(
                rule_type=RuleType.GREATER_THAN,
                features=["AIT402"],
                params={"threshold": 0.35},
                weight=1.5,
                description=(
                    "AIT402 > 0.35 (scaled p0.999 of normal): "
                    "RO conductivity spike → attack. "
                    "Normal mean_sc=0.110 (p0.999=0.35), attack mean_sc=0.587. Sep=0.23."
                )
            ))
        """
        # ══════════════════════════════════════════════════════════════════════
        # GROUP 5 — Actuator state  (scale-invariant binary rule)
        # ══════════════════════════════════════════════════════════════════════

        # P204 == P206 → fault
        # Binary {0,1} sensors; MinMax maps {0,1}→{0,1} exactly.
        # Normal: complementary (diff≈1.0) → residual=clamp(0.5-1.0)=0.
        # Attack: same state (diff≈0.0) → residual=clamp(0.5-0.0)=0.5.
        if all(f in self.name_to_idx for f in ["P204", "P206"]):
            self.rules.append(PhysicsRule(
                rule_type=RuleType.EQUALITY,
                features=["P204", "P206"],
                params={"tolerance": 0.5},
                weight=1.0,
                description=(
                    "P204 == P206: complementary pump signals in same state → fault. "
                    "Binary {0,1}: normal diff=1.0→res=0; attack diff=0.0→res=0.5."
                )
            ))
        """


    def _build_wadi_rules(self):
        def has_all(features):
            return all(f in self.name_to_idx for f in features)

        range_rules = [
            # 1_AIT_001_PV: normal mostly around 156–178, max around 214.
            # Zero is suspicious for this sensor.
            (
                RuleType.LESS_THAN,
                ["1_AIT_001_PV"],
                {"threshold": 100.0},
                0.7,
                "1_AIT_001_PV abnormally low"
            ),
            (
                RuleType.GREATER_THAN,
                ["1_AIT_001_PV"],
                {"threshold": 220.0},
                0.7,
                "1_AIT_001_PV abnormally high"
            ),

            # 1_AIT_002_PV: normal around 0.58–0.66.
            # Values near 0 or above 1 are suspicious.
            (
                RuleType.LESS_THAN,
                ["1_AIT_002_PV"],
                {"threshold": 0.20},
                0.7,
                "1_AIT_002_PV abnormally low"
            ),
            (
                RuleType.GREATER_THAN,
                ["1_AIT_002_PV"],
                {"threshold": 1.00},
                0.7,
                "1_AIT_002_PV abnormally high"
            ),

            # 2_LT_002_PV: normal level around 65.7–83.0.
            # Use slightly wider bounds than observed range.
            (
                RuleType.LESS_THAN,
                ["2_LT_002_PV"],
                {"threshold": 64.0},
                0.9,
                "2_LT_002_PV tank level abnormally low"
            ),
            (
                RuleType.GREATER_THAN,
                ["2_LT_002_PV"],
                {"threshold": 85.0},
                0.9,
                "2_LT_002_PV tank level abnormally high"
            ),

            # 2B_AIT_002_PV: normal around 9.0–9.2.
            # Observed max 38.783 is very suspicious.
            (
                RuleType.LESS_THAN,
                ["2B_AIT_002_PV"],
                {"threshold": 8.3},
                0.8,
                "2B_AIT_002_PV abnormally low"
            ),
            (
                RuleType.GREATER_THAN,
                ["2B_AIT_002_PV"],
                {"threshold": 10.0},
                0.9,
                "2B_AIT_002_PV abnormally high"
            ),

            # 1_FIT_001_PV: observed max around 2.066.
            (
                RuleType.GREATER_THAN,
                ["1_FIT_001_PV"],
                {"threshold": 2.20},
                0.6,
                "1_FIT_001_PV flow abnormally high"
            ),

            # 2_FIT_001_PV: observed max around 2.302.
            (
                RuleType.GREATER_THAN,
                ["2_FIT_001_PV"],
                {"threshold": 2.50},
                0.6,
                "2_FIT_001_PV flow abnormally high"
            ),

            # 2_FIC_501_PV: usually low, Q75 around 0.128, but max 2.590.
            (
                RuleType.GREATER_THAN,
                ["2_FIC_501_PV"],
                {"threshold": 0.50},
                0.7,
                "2_FIC_501_PV flow/controller PV abnormally high"
            ),
        ]

        for rule_type, features, params, weight, description in range_rules:
            if has_all(features):
                self.rules.append(PhysicsRule(
                    rule_type=rule_type,
                    features=features,
                    params=params,
                    weight=weight,
                    description=description
                ))

        actuator_rules = [
            (
                RuleType.LESS_THAN,
                ["1_MV_002_STATUS"],
                {"threshold": 0.5},
                0.6,
                "1_MV_002_STATUS unexpectedly not open"
            ),
            (
                RuleType.LESS_THAN,
                ["1_MV_003_STATUS"],
                {"threshold": 0.5},
                0.6,
                "1_MV_003_STATUS unexpectedly not open"
            ),
            (
                RuleType.LESS_THAN,
                ["1_P_006_STATUS"],
                {"threshold": 0.5},
                0.6,
                "1_P_006_STATUS unexpectedly off"
            ),
            (
                RuleType.GREATER_THAN,
                ["2_MCV_007_CO"],
                {"threshold": 1.0},
                0.6,
                "2_MCV_007_CO unexpectedly open/nonzero"
            ),
        ]

        for rule_type, features, params, weight, description in actuator_rules:
            if has_all(features):
                self.rules.append(PhysicsRule(
                    rule_type=rule_type,
                    features=features,
                    params=params,
                    weight=weight,
                    description=description
                ))

        valve_flow_pairs = [
            # Stage 1 motorized valves associated with 1_FIT_001_PV.
            (
                "1_MV_002_STATUS",
                "1_FIT_001_PV",
                0.05,
                "1_MV_002_STATUS closed but 1_FIT_001_PV has flow"
            ),
            (
                "1_MV_003_STATUS",
                "1_FIT_001_PV",
                0.05,
                "1_MV_003_STATUS closed but 1_FIT_001_PV has flow"
            ),

            # 2_MCV_007_CO is command/output and is always 0 in your data.
            # Since 2_FIC_501_PV has small normal values, use a larger threshold
            # to avoid flagging tiny residual/measurement noise.
            (
                "2_MCV_007_CO",
                "2_FIC_501_PV",
                0.50,
                "2_MCV_007_CO closed but 2_FIC_501_PV is high"
            ),
        ]

        for valve, flow, flow_threshold, description in valve_flow_pairs:
            if has_all([valve, flow]):
                self.rules.append(PhysicsRule(
                    rule_type=RuleType.VALVE_FLOW,
                    features=[valve, flow],
                    params={
                        "closed_state": 0,
                        "open_state": 1,
                        "flow_threshold": flow_threshold
                    },
                    weight=0.8,
                    description=description
                ))

        if has_all(["1_P_006_STATUS", "1_FIT_001_PV"]):
            self.rules.append(PhysicsRule(
                rule_type=RuleType.PUMP_FLOW,
                features=["1_P_006_STATUS", "1_FIT_001_PV"],
                params={
                    "running_state": 1,
                    "off_state": 0,
                    "flow_threshold": 0.001
                },
                weight=0.5,
                description="1_P_006_STATUS running should be consistent with 1_FIT_001_PV"
            ))

        temporal_rules = [
            (
                ["2_LT_002_PV"],
                {"max_rate": 1.0},
                0.8,
                "2_LT_002_PV tank level should change smoothly"
            ),
            (
                ["1_AIT_001_PV"],
                {"max_rate": 10.0},
                0.5,
                "1_AIT_001_PV should change smoothly"
            ),
            (
                ["1_AIT_002_PV"],
                {"max_rate": 0.20},
                0.5,
                "1_AIT_002_PV should change smoothly"
            ),
            (
                ["2B_AIT_002_PV"],
                {"max_rate": 0.30},
                0.7,
                "2B_AIT_002_PV should change smoothly"
            ),
            (
                ["1_FIT_001_PV"],
                {"max_rate": 1.0},
                0.5,
                "1_FIT_001_PV flow should change smoothly"
            ),
            (
                ["2_FIT_001_PV"],
                {"max_rate": 1.0},
                0.5,
                "2_FIT_001_PV flow should change smoothly"
            ),
            (
                ["2_FIC_501_PV"],
                {"max_rate": 0.30},
                0.5,
                "2_FIC_501_PV should change smoothly"
            ),
        ]

        for features, params, weight, description in temporal_rules:
            if has_all(features):
                self.rules.append(PhysicsRule(
                    rule_type=RuleType.TEMPORAL_RATE,
                    features=features,
                    params=params,
                    weight=weight,
                    description=description
                ))

        if has_all(["2_LT_002_PV", "2_FIT_001_PV"]):
            self.rules.append(PhysicsRule(
                rule_type=RuleType.MASS_BALANCE,
                features=["2_LT_002_PV", "2_FIT_001_PV"],
                params={
                    "tank_area": 1.0,
                    "dt": 1.0,
                    "tolerance": 2.0
                },
                weight=0.5,
                description="Approximate mass balance for 2_LT_002_PV using 2_FIT_001_PV"
            ))


    def _build_wadi_rules_scaled(self):
        """
        WADI physics rules for MinMax-scaled data [0, 1].

        x_sc = (x_raw - n_min) / (n_max - n_min).
        Scaler MUST be fit on clean normal training data only.

        Threshold derivation (from wadi_analysis.csv):

          GREATER_THAN → p0.999 of normal in scaled space
                         = n_mean_sc + 3.09 * n_std_sc
                         If p0.999 > 1.0 (large normal std), capped at 1.0.

          LESS_THAN    → p0.001 of normal in scaled space
                         = n_mean_sc - 3.09 * n_std_sc
                         If p0.001 < 0 (normal extends below n_min),
                         the rule is removed (no valid threshold).

          PUMP_FLOW    → scale-invariant: binary {0,1} status sensors
                         map to {0,1} under MinMax.

        Removed sensors:
          2_PIC_003_PV / 2_PIT_003_PV: attack mean (sc≈0.10) > normal mean
            (sc≈0.054) — attacks raise pressure, not lower it. p0.001 is
            negative (normal extends below n_min). No consistent threshold
            direction exists.
        """

        def has(features):
            return all(f in self.name_to_idx for f in features)

        # ── GREATER_THAN ──────────────────────────────────────────────────────

        # 1_AIT_001_PV: n_mean_sc=0.797, n_std_sc=0.065 → p0.999_sc=0.997
        # a_max_sc=2.961 → fires on attack peaks. a_mean_sc=0.878 (below thr).
        if has(["1_AIT_001_PV"]):
            self.rules.append(PhysicsRule(
                rule_type=RuleType.GREATER_THAN,
                features=["1_AIT_001_PV"],
                params={"threshold": 0.997},
                weight=1.5,
                description=(
                    "1_AIT_001_PV > 0.997 (p0.999_sc): quality spike → attack. "
                    "a_max_sc=2.961 fires well above this."
                )
            ))

        # 1_FIT_001_PV: n_std_sc=0.410 → p0.999>1.0, capped at 1.0.
        # a_max_sc=1.207 fires at peaks.
        if has(["1_FIT_001_PV"]):
            self.rules.append(PhysicsRule(
                rule_type=RuleType.GREATER_THAN,
                features=["1_FIT_001_PV"],
                params={"threshold": 1.0},
                weight=1.2,
                description=(
                    "1_FIT_001_PV > 1.0 (scaled n_max, p0.999>1 capped): "
                    "abnormally high flow → attack. a_max_sc=1.207."
                )
            ))

        # 2_FIT_001_PV: n_std_sc=0.286 → p0.999>1.0, capped at 1.0.
        # a_max_sc=1.121 fires at peaks.
        if has(["2_FIT_001_PV"]):
            self.rules.append(PhysicsRule(
                rule_type=RuleType.GREATER_THAN,
                features=["2_FIT_001_PV"],
                params={"threshold": 1.0},
                weight=1.0,
                description=(
                    "2_FIT_001_PV > 1.0 (scaled n_max, p0.999>1 capped): "
                    "high flow → attack. a_max_sc=1.121."
                )
            ))

        # 2_FIT_002_PV: n_mean_sc=0.179, n_std_sc=0.169 → p0.999_sc=0.701
        # a_max_sc=1.050 fires above 0.70.
        if has(["2_FIT_002_PV"]):
            self.rules.append(PhysicsRule(
                rule_type=RuleType.GREATER_THAN,
                features=["2_FIT_002_PV"],
                params={"threshold": 0.70},
                weight=1.0,
                description=(
                    "2_FIT_002_PV > 0.70 (p0.999_sc=0.701): high flow → attack. "
                    "a_max_sc=1.050 fires above this."
                )
            ))

        # 1_P_006_STATUS: state 2 (bi-speed) scaled to 1.0; threshold=0.75=midpoint.
        if has(["1_P_006_STATUS"]):
            self.rules.append(PhysicsRule(
                rule_type=RuleType.GREATER_THAN,
                features=["1_P_006_STATUS"],
                params={"threshold": 0.75},
                weight=0.8,
                description=(
                    "1_P_006_STATUS > 0.75 (scaled midpoint): pump bi-speed state → attack. "
                    "State 1=0.5 sc (normal), state 2=1.0 sc (attack)."
                )
            ))

        # ── LESS_THAN ─────────────────────────────────────────────────────────

        # 2_LT_002_PV: n_mean_sc=0.745, n_std_sc=0.049 → p0.001_sc=0.593
        # a_mean_sc=0.678 > 0.593 (weak mean separation), but a_min_sc=-0.006
        # catches extreme drops. Low weight.
        if has(["2_LT_002_PV"]):
            self.rules.append(PhysicsRule(
                rule_type=RuleType.LESS_THAN,
                features=["2_LT_002_PV"],
                params={"threshold": 0.59},
                weight=0.8,
                description=(
                    "2_LT_002_PV < 0.59 (p0.001_sc): tank level critically low. "
                    "Catches extreme drops; a_mean_sc=0.678 (weak mean sep). Low weight."
                )
            ))

        # 2_PIC_003_PV and 2_PIT_003_PV EXCLUDED:
        # p0.001_sc < 0 (normal extends below n_min); attack mean > normal mean.
        # No valid threshold in either direction without unacceptable FP rate.

        # ── PUMP_FLOW (scale-invariant) ────────────────────────────────────────

        # 1_P_002_STATUS: pump ON (scaled=1.0) but flow < 0.001 (near scaled n_min)
        if has(["1_P_002_STATUS", "1_FIT_001_PV"]):
            self.rules.append(PhysicsRule(
                rule_type=RuleType.PUMP_FLOW,
                features=["1_P_002_STATUS", "1_FIT_001_PV"],
                params={"running_state": 1, "off_state": 0, "flow_threshold": 0.001},
                weight=0.9,
                description=(
                    "1_P_002_STATUS=1 but 1_FIT_001_PV < 0.001 (scaled): "
                    "pump on, no flow. Scale-invariant binary status."
                )
            ))

        # 1_P_004_STATUS: redundant pump pair
        if has(["1_P_004_STATUS", "1_FIT_001_PV"]):
            self.rules.append(PhysicsRule(
                rule_type=RuleType.PUMP_FLOW,
                features=["1_P_004_STATUS", "1_FIT_001_PV"],
                params={"running_state": 1, "off_state": 0, "flow_threshold": 0.001},
                weight=0.9,
                description=(
                    "1_P_004_STATUS=1 but 1_FIT_001_PV < 0.001 (scaled): "
                    "pump on, no flow. Redundant pair with P002 rule."
                )
            ))

    def get_rule_indices(self, rule_type: RuleType) -> List[int]:
        """Return indices of rules matching a type."""
        return [i for i, r in enumerate(self.rules) if r.rule_type == rule_type]

    def __len__(self):
        return len(self.rules)

    def __repr__(self):
        return (f"PhysicsRuleSet(plant={self.plant}, n_rules={len(self.rules)}, "
                f"n_features={len(self.feature_names)})")


class PhysicsResidualCalculator:
    """
    Compute residuals for each physics rule type.
    All residuals normalized by training-set std before loss.
    """

    def __init__(self, rule_set: PhysicsRuleSet, feature_names: List[str],
                 residual_stds: Dict[str, float] = None):
        """
        Parameters
        ----------
        rule_set : PhysicsRuleSet
        feature_names : list of str
        residual_stds : dict
            Pre-computed std of each residual type on clean training data.
            Keys: "mass_balance", "valve_flow", "pump_flow", "equality", etc.
            If None, defaults to 1.0 (no normalization).
        """
        self.rule_set = rule_set
        self.feature_names = feature_names
        self.name_to_idx = {name: i for i, name in enumerate(feature_names)}
        self.residual_stds = residual_stds or {}

    def _get_std(self, residual_type: str) -> float:
        """Fetch pre-computed std or default to 1.0."""
        return self.residual_stds.get(residual_type, 1.0)

    def compute_all_residuals(self, x_t: torch.Tensor, x_next: torch.Tensor,
                              x_prev: torch.Tensor = None,
                              state: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        Compute residuals for all rules.

        Parameters
        ----------
        x_t : (batch, n_features)
            Current timestep — the "from" state of two-step rules
            (MASS_BALANCE, TEMPORAL_RATE).
        x_next : (batch, n_features)
            Next timestep — the "to" state of two-step rules.
        x_prev : (batch, n_features), optional
            Reserved; unused by the current rule set.
        state : (batch, n_features), optional
            State on which SINGLE-timestep rules (valve, pump, equality,
            greater/less-than) are evaluated.  Defaults to `x_t`.

            This argument exists so the composite loss can evaluate every rule
            on the model's PREDICTION.  With state=None the single-step rules
            read `x_t`, which in the loss is observed data — a constant with
            respect to the model parameters, so the physics term would have no
            gradient at all.  Passing state=x_pred (and x_next=x_pred) makes
            every rule type differentiable.  Detection-time scoring keeps the
            default so residuals are measured on observations.

        Returns
        -------
        residuals : dict
            Keys: "rule_{i}".  Values: (batch,) tensors.
        """
        residuals = {}
        if state is None:
            state = x_t

        for i, rule in enumerate(self.rule_set.rules):
            if rule.rule_type == RuleType.MASS_BALANCE:
                res = self._mass_balance_residual(rule, x_t, x_next)
            elif rule.rule_type == RuleType.VALVE_FLOW:
                res = self._valve_flow_residual(rule, state)
            elif rule.rule_type == RuleType.PUMP_FLOW:
                res = self._pump_flow_residual(rule, state)
            elif rule.rule_type == RuleType.EQUALITY:
                res = self._equality_residual(rule, state)
            elif rule.rule_type == RuleType.GREATER_THAN:
                res = self._greater_than_residual(rule, state)
            elif rule.rule_type == RuleType.LESS_THAN:
                res = self._less_than_residual(rule, state)
            elif rule.rule_type == RuleType.TEMPORAL_RATE:
                res = self._temporal_rate_residual(rule, x_t, x_next)
            else:
                continue

            # Apply rule weight, then normalise by pre-computed std.
            # rule.weight was previously ignored — all rules contributed equally.
            # Now weights (1.2, 1.5, 0.8 …) scale each rule's contribution
            # so high-confidence rules (e.g. AIT402 conductivity spike) have
            # stronger gradient signal than marginal ones (e.g. LIT401 level).
            std = self._get_std(rule.rule_type.value)
            res = (res * rule.weight) / (std + 1e-8)
            residuals[f"rule_{i}"] = res

        return residuals

    def _mass_balance_residual(self, rule: PhysicsRule, x_t: torch.Tensor,
                               x_next: torch.Tensor) -> torch.Tensor:
        """
        Eq. 1: A * (L[t+1] - L[t]) / Δt = Q_in[t] - Q_out[t]
        Residual: |A * ΔL / Δt - (Q_in - Q_out)|
        """
        features = rule.features  # [level, flow_in, flow_out]
        if len(features) != 3:
            return torch.zeros(x_t.shape[0], device=x_t.device)

        level_idx = self.name_to_idx[features[0]]
        fin_idx = self.name_to_idx[features[1]]
        fout_idx = self.name_to_idx[features[2]]

        A = rule.params.get("tank_area", 1.0)
        dt = rule.params.get("dt", 1.0)

        # Level change
        delta_L = x_next[:, level_idx] - x_t[:, level_idx]
        lhs = A * delta_L / dt

        # Net flow
        rhs = x_t[:, fin_idx] - x_t[:, fout_idx]

        residual = torch.abs(lhs - rhs)
        return residual

    def _valve_flow_residual(self, rule: PhysicsRule, x_t: torch.Tensor) -> torch.Tensor:
        """
        Eq. 2: (1 - valve_state) * |flow| ≈ 0
        Residual: (1 - valve) * |flow|
        """
        features = rule.features  # [valve, flow]
        if len(features) != 2:
            return torch.zeros(x_t.shape[0], device=x_t.device)

        valve_idx = self.name_to_idx[features[0]]
        flow_idx = self.name_to_idx[features[1]]

        valve_state = x_t[:, valve_idx]  # 0 or 1
        flow = x_t[:, flow_idx]

        residual = (1.0 - valve_state) * torch.abs(flow)
        return residual

    def _pump_flow_residual(self, rule: PhysicsRule, x_t: torch.Tensor) -> torch.Tensor:
        """
        Eq. 3: (1-pump)*|flow| + pump*max(0, τ - flow)
        Residual penalizes:
        - Pump OFF but flow present
        - Pump ON but flow below threshold τ

        BUG FIX (param key):
          Rules created by _build_swat_rules_scaled / _build_wadi_rules_scaled
          store the flow threshold under "flow_threshold".
          The original code looked for "pump_threshold" (default=2.0), which
          was never found → tau=2.0 >> any MinMax-scaled flow ∈ [0,1] →
          term2 ≈ 1.7–2.0 on EVERY sample, inflating all scores uniformly
          and collapsing precision to the attack base rate.
          FIX: try "flow_threshold" first, then "pump_threshold", default=0.1.
          0.1 is a safe default for scaled data (10% of range).
        """
        features = rule.features  # [pump, flow]
        if len(features) != 2:
            return torch.zeros(x_t.shape[0], device=x_t.device)

        pump_idx = self.name_to_idx[features[0]]
        flow_idx = self.name_to_idx[features[1]]

        # Accept both key names; default 0.1 is safe for MinMax-scaled data
        tau = rule.params.get(
            "flow_threshold",
            rule.params.get("pump_threshold", 0.1)
        )

        pump_state = x_t[:, pump_idx]
        flow = x_t[:, flow_idx]

        # Term 1: pump OFF but flow present
        term1 = (1.0 - pump_state) * torch.abs(flow)

        # Term 2: pump ON but flow below threshold
        term2 = pump_state * torch.clamp(tau - flow, min=0.0)

        residual = term1 + term2
        return residual

    def _equality_residual(self, rule: PhysicsRule, x_t: torch.Tensor) -> torch.Tensor:
        features = rule.features
        if len(features) != 2:
            return torch.zeros(x_t.shape[0], device=x_t.device)

        idx1, idx2 = self.name_to_idx[features[0]], self.name_to_idx[features[1]]
        diff = torch.abs(x_t[:, idx1] - x_t[:, idx2])
        tolerance = rule.params.get("tolerance", 0.5)

        # Anomaly when diff < tolerance
        residual = torch.clamp(tolerance - diff, min=0.0)
        return residual

    def _greater_than_residual(self, rule: PhysicsRule, x_t: torch.Tensor) -> torch.Tensor:
        """
        Feature should NOT exceed threshold.
        Residual: max(0, x - threshold)
        """
        features = rule.features
        if len(features) != 1:
            return torch.zeros(x_t.shape[0], device=x_t.device)

        idx = self.name_to_idx[features[0]]
        threshold = rule.params.get("threshold", 0.0)

        residual = torch.clamp(x_t[:, idx] - threshold, min=0.0)
        return residual

    def _less_than_residual(self, rule: PhysicsRule, x_t: torch.Tensor) -> torch.Tensor:
        """
        Feature should NOT go below threshold.
        Residual: max(0, threshold - x)
        """
        features = rule.features
        if len(features) != 1:
            return torch.zeros(x_t.shape[0], device=x_t.device)

        idx = self.name_to_idx[features[0]]
        threshold = rule.params.get("threshold", 0.0)

        residual = torch.clamp(threshold - x_t[:, idx], min=0.0)
        return residual

    def _temporal_rate_residual(self, rule: PhysicsRule, x_t: torch.Tensor,
                                x_next: torch.Tensor) -> torch.Tensor:
        """
        Eq. 6: |x[t+1] - x[t]| ≤ max_rate
        Residual: max(0, |Δx| - max_rate)
        """
        features = rule.features
        if len(features) != 1:
            return torch.zeros(x_t.shape[0], device=x_t.device)

        idx = self.name_to_idx[features[0]]
        max_rate = rule.params.get("max_rate", 0.5)

        delta = torch.abs(x_next[:, idx] - x_t[:, idx])
        residual = torch.clamp(delta - max_rate, min=0.0)
        return residual


class TemporalConsistencyLoss(nn.Module):
    """
    Physics-aware temporal smoothness for SWaT/WADI.
    Penalizes unrealistic rate-of-change per feature type.
    """

    def __init__(self, feature_names: List[str], dt: float = 1.0):
        super().__init__()
        self.feature_names = feature_names
        self.dt = dt

        # Feature-specific max rates (cm/s, L/min, bar/s, etc.)
        self.max_rates = self._init_max_rates()
        self.feature_scales = self._init_scales()

    def _init_max_rates(self) -> Dict[str, float]:
        """
        Max rate-of-change per feature type in SCALED [0,1] space.

        BUG FIX: original rates (0.5 cm/s, 10 L/min, 50 bar/s) were in
        raw engineering units. After MinMax scaling consecutive-step deltas
        are ≈ 0.001–0.01, so clamp(|delta| - 10.0, min=0) = 0 ALWAYS →
        temporal loss was a permanent no-op.

        Rates below are in scaled units, derived as:
            max_rate_scaled = max_rate_raw / sensor_range
        Representative sensor ranges (from SWaT/WADI training data):
            LIT: range ≈ 200 cm   → 0.5/200  ≈ 0.0025 → use 0.005 (small buffer)
            FIT: range ≈ 5 L/min  → 1.0/5    ≈ 0.20   → use 0.20
            PIT: range ≈ 10 bar   → 0.5/10   ≈ 0.05   → use 0.05
            AIT: range ≈ 300 µS   → 5/300    ≈ 0.017  → use 0.02
        These are conservative upper bounds — choose smaller to penalise
        faster changes more aggressively.
        """
        rates = {}
        for fname in self.feature_names:
            if "LIT" in fname or "_LT_" in fname:   # Level — slow, smooth
                rates[fname] = 0.005
            elif "FIT" in fname or "_FIT_" in fname:  # Flow — can change faster
                rates[fname] = 0.20
            elif "PIT" in fname or "_PIT_" in fname:  # Pressure
                rates[fname] = 0.05
            elif "AIT" in fname or "_AT_" in fname:   # Conductivity/chemistry
                rates[fname] = 0.02
            elif fname.startswith("MV") or fname.startswith("UV"):
                rates[fname] = float('inf')  # Discrete actuator — skip
            elif fname.startswith("P") and not fname.startswith("PIT"):
                rates[fname] = float('inf')  # Pump/actuator — discrete
            else:
                rates[fname] = 0.10   # Default: 10% of range per timestep
        return rates

    def _init_scales(self) -> Dict[str, float]:
        """Normalization scales (typical sensor ranges)."""
        scales = {}
        for fname in self.feature_names:
            if "LIT" in fname or "_LT_" in fname:
                scales[fname] = 100.0  # 0–100 cm
            elif "FIT" in fname or "_FIT_" in fname:
                scales[fname] = 50.0  # 0–50 L/min
            elif fname.startswith("P"):
                scales[fname] = 10.0  # 0–10 bar
            elif "AIT" in fname or "_AT_" in fname:
                scales[fname] = 1000.0  # 0–1000 µS/cm
            else:
                scales[fname] = 1.0
        return scales

    def forward(
            self,
            x_pred: torch.Tensor,  # [batch, n_features]
            x_prev: torch.Tensor,  # [batch, n_features]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute temporal smoothness penalty, bounded to (0, 1) via tanh.

        FIX 1 — divisor bug:
            The original code divided by (self.dt * 1e-3), so with dt=1.0 the
            divisor was 0.001, amplifying every delta by 1000× BEFORE squaring.
            A scaled delta of 0.01 became rate=10 → excess²=100, dwarfing MSE.
            Fixed: divide by self.dt directly (dt=1.0 → no amplification).

        FIX 2 — unbounded output:
            excess² is unbounded. A single large delta can produce a loss
            of hundreds, overwhelming the MSE term regardless of the weight.
            Fixed: apply tanh(total_loss) before returning, capping output
            to (0, 1). The gradient near 0 is 1 (sensitive to small violations)
            and saturates for very large violations — same design as physics.

        FIX 3 — double compression:
            This used to return tanh(raw_loss), and CompositeLoss then applied
            tanh() a second time.  The composite term was therefore capped at
            tanh(tanh(x)) < 0.76 with a flattened gradient.  The RAW penalty is
            returned now; CompositeLoss applies the single tanh with its own
            `temporal_scale`.

        Returns
        -------
        loss   : scalar >= 0, uncompressed
        details: dict with per-feature contributions for logging
        """
        batch_size, n_features = x_pred.shape

        # Rate of change per timestep — no 1e-3 amplification
        delta = x_pred - x_prev          # [batch, n_features]
        rate  = delta / self.dt          # FIX 1: was delta / (self.dt * 1e-3)

        losses_per_feature = []
        details = {}

        for i, fname in enumerate(self.feature_names):
            max_rate = self.max_rates.get(fname, 0.10)  # 10% of range per step

            if max_rate == float('inf'):   # discrete actuator — skip
                continue

            excess = torch.clamp(torch.abs(rate[:, i]) - max_rate, min=0.0)
            feature_loss = torch.mean(excess ** 2)
            losses_per_feature.append(feature_loss)
            details[fname] = feature_loss.item()   # raw value for logging

        if not losses_per_feature:
            return torch.tensor(0.0, device=x_pred.device), details

        # Raw (uncompressed) penalty — CompositeLoss applies tanh once, with
        # its own temporal_scale, exactly as it does for the physics term.
        raw_loss = torch.stack(losses_per_feature).mean()
        return raw_loss, details


class CompositeLoss(nn.Module):
    """
    Physics-informed composite loss.

        total = UW( [ recon, w_p * physics, w_t * temporal, w_g * graph ] )

    Every auxiliary term is (a) compressed to (0, 1) with tanh so a single
    large residual cannot swamp the reconstruction term, then (b) rescaled to
    the current magnitude of the reconstruction loss, so each weight means
    "this term contributes ~w x the reconstruction loss" rather than an
    absolute magnitude that depends on the dataset's units.

    Which terms are active is decided by `ablation_mode` ALONE, so `n_tasks`
    always equals the number of entries appended to `losses` in forward():

        "mse_only"      -> [recon]
        "mse+physics"   -> [recon, physics]
        "mse+temporal"  -> [recon, temporal]
        "mse+graph"     -> [recon, graph]
        "full"          -> [recon, physics, temporal, graph]

    Weights control MAGNITUDE, never activation.  Passing graph_weight>0 to an
    "mse+physics" run must not silently add a fourth task and desynchronise
    the uncertainty module.
    """

    def __init__(
            self,
            residual_calculator,
            feature_names: List[str],
            uncertainty_weighting: str = "kendall",
            ablation_mode: str = "full",
            physics_weight: float = 0.1,
            temporal_weight: float = 0.05,
            graph_weight: float = 0.05,
            min_physics_weight: float = 0.05,
            use_huber: bool = True,
            huber_delta: float = 1.0,
            physics_scale: float = 1.0,   # tanh half-saturation for physics
            temporal_scale: float = 1.0,  # tanh half-saturation for temporal
            graph_scale: float = 1.0,     # tanh half-saturation for graph
            physics_on_prediction: bool = True,
            graph_ignore_self_loops: bool = True,
    ):
        """
        physics_on_prediction : bool
            True  — physics rules are evaluated on the model's prediction, so
                    the physics term has a gradient (a physics-INFORMED loss).
            False — legacy behaviour: rules are evaluated on observed data
                    only.  The term is then a constant w.r.t. the model
                    parameters and contributes NO gradient; "MSE + Physics"
                    becomes indistinguishable from "MSE only" apart from the
                    uncertainty weights.  Kept only for reproducing old runs.

        graph_ignore_self_loops : bool
            The prior adjacency from pre_pipeline has self-loops added.  A
            self-loop contributes 0 to the smoothness numerator but 1 to the
            edge count, diluting the term by (E + d) / E.  True excludes the
            diagonal from both.
        """
        super().__init__()
        self.residual_calc = residual_calculator
        self.feature_names = feature_names
        self.ablation_mode = ablation_mode
        self.physics_weight = physics_weight
        self.temporal_weight = temporal_weight
        self.graph_weight = graph_weight
        self.min_physics_weight = min_physics_weight
        self.use_huber = use_huber
        self.huber_delta = huber_delta
        self.physics_scale = physics_scale
        self.temporal_scale = temporal_scale
        self.graph_scale = graph_scale
        self.physics_on_prediction = physics_on_prediction
        self.graph_ignore_self_loops = graph_ignore_self_loops

        self.use_physics = ("physics" in ablation_mode) or (ablation_mode == "full")
        self.use_temporal = ("temporal" in ablation_mode) or (ablation_mode == "full")
        self.use_graph = ("graph" in ablation_mode) or (ablation_mode == "full")

        n_tasks = 1 + int(self.use_physics) + int(self.use_temporal) + int(self.use_graph)

        # A term named by the ablation mode must actually fire, even if the
        # caller left its weight at zero.
        if self.use_graph and self.graph_weight <= 0:
            self.graph_weight = 0.05
        if self.use_temporal and self.temporal_weight <= 0:
            self.temporal_weight = 0.05

        if uncertainty_weighting == "kendall":
            self.uncertainty = KendallGalUncertainty(n_tasks)
        elif uncertainty_weighting == "gradnorm":
            self.uncertainty = GradNormWeighting(n_tasks)
        elif uncertainty_weighting == "dwa":
            self.uncertainty = DynamicWeightAveraging(n_tasks)
        else:
            self.uncertainty = None

        self.n_tasks = n_tasks
        self.temporal_loss_fn = TemporalConsistencyLoss(feature_names, dt=1.0)
        self.reset_history()

    @staticmethod
    def _compress(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        """
        Compress a non-negative tensor to (0, 1) via tanh(x / scale).

          x = 0      -> 0            (no violation)
          x = scale  -> tanh(1) ~ 0.76
          x -> inf   -> 1            (hard cap)
          d/dx at 0  -> 1 / scale    (small violations stay sensitive)
        """
        return torch.tanh(x / scale)

    @staticmethod
    def _anchor(compressed: torch.Tensor, mse_ref: torch.Tensor,
                cap: float = 10.0) -> torch.Tensor:
        """
        Rescale a compressed term to the current magnitude of the
        reconstruction loss.  The normalisation factor is detached, so the
        gradient flows only through `compressed` and the two tasks stay
        cleanly separated.  `cap` bounds the term at cap x recon.
        """
        factor = (mse_ref / (compressed.detach() + 1e-8)).clamp(max=cap)
        return compressed * factor

    def forward(
            self,
            x_pred: torch.Tensor,
            x_true: torch.Tensor,
            x_prev: torch.Tensor = None,
            adjacency: torch.Tensor = None,
            return_components: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """
        x_pred    : (batch, d) predicted next timestep
        x_true    : (batch, d) observed next timestep
        x_prev    : (batch, d) last observed timestep of the input window
        adjacency : (d, d) prior graph — REQUIRED whenever the graph term is
                    active, otherwise the term degenerates to a constant zero.
        """
        device = x_pred.device
        zero = torch.zeros((), device=device)
        losses: List[torch.Tensor] = []
        components: Dict[str, float] = {}

        # ── 1. Reconstruction — sets the scale baseline ───────────────────────
        if self.use_huber:
            recon_loss = F.smooth_l1_loss(x_pred, x_true, beta=self.huber_delta)
        else:
            recon_loss = F.mse_loss(x_pred, x_true)
        losses.append(recon_loss)
        components["mse"] = recon_loss.item()
        self.loss_history["mse"].append(recon_loss.item())

        # Stable, detached reference for rescaling the auxiliary terms.
        # Clamped to avoid divide-by-zero and runaway scaling early in training.
        mse_ref = recon_loss.detach().clamp(min=1e-6, max=1.0)

        # ── 2. Physics ────────────────────────────────────────────────────────
        if self.use_physics:
            physics_raw, physics_loss = zero, zero
            has_rules = len(self.residual_calc.rule_set.rules) > 0
            if x_prev is not None and has_rules:
                if self.physics_on_prediction:
                    # Rules evaluated on the PREDICTION -> differentiable.
                    residuals = self.residual_calc.compute_all_residuals(
                        x_t=x_prev, x_next=x_pred, state=x_pred
                    )
                else:
                    residuals = self.residual_calc.compute_all_residuals(
                        x_t=x_prev, x_next=x_true
                    )
                if residuals:
                    rule_means = torch.stack(list(residuals.values())).mean(dim=1)
                    physics_raw = rule_means.mean()
                    compressed = self._compress(rule_means, self.physics_scale).mean()
                    physics_loss = self._anchor(compressed, mse_ref)

            eff_pw = max(self.physics_weight, self.min_physics_weight)
            losses.append(eff_pw * physics_loss)
            components["physics"] = physics_loss.item()
            components["physics_raw"] = physics_raw.item()
            self.loss_history["physics"].append(physics_loss.item())
            self.loss_history["physics_raw"].append(physics_raw.item())

        # ── 3. Temporal consistency ───────────────────────────────────────────
        # Appended unconditionally when active, so len(losses) == n_tasks even
        # if x_prev is missing; otherwise the uncertainty module is silently
        # skipped and the run is no longer the configuration it claims to be.
        if self.use_temporal:
            temporal_raw, temporal_loss = zero, zero
            if x_prev is not None:
                temporal_raw, temporal_details = self.temporal_loss_fn(x_pred, x_prev)
                compressed = self._compress(temporal_raw, self.temporal_scale)
                temporal_loss = self._anchor(compressed, mse_ref)
                components.update({f"temp_{k}": v
                                   for k, v in temporal_details.items()})

            losses.append(self.temporal_weight * temporal_loss)
            components["temporal"] = temporal_loss.item()
            components["temporal_raw"] = temporal_raw.item()
            self.loss_history["temporal"].append(temporal_loss.item())
            self.loss_history["temporal_raw"].append(temporal_raw.item())

        # ── 4. Graph regularisation ───────────────────────────────────────────
        if self.use_graph:
            graph_raw, graph_loss = zero, zero
            if adjacency is not None:
                if adjacency.device != device:
                    adjacency = adjacency.to(device)
                if not self._warned_no_edges:
                    off_diag = adjacency.numel() - adjacency.diagonal().numel()
                    n_edges = int((adjacency != 0).sum().item()
                                  - (adjacency.diagonal() != 0).sum().item())
                    if off_diag > 0 and n_edges == 0:
                        warnings.warn(
                            "CompositeLoss: graph term is active but the "
                            "adjacency has no off-diagonal edges (identity?). "
                            "The smoothness term is identically zero, so the "
                            "'graph' ablation arm is indistinguishable from "
                            "the baseline. Pass the prior graph from "
                            "pre_pipeline()['prior_adjacency'].",
                            RuntimeWarning,
                        )
                    self._warned_no_edges = True
                graph_raw = self._graph_regularization(x_pred, adjacency)
                compressed = self._compress(graph_raw, self.graph_scale)
                graph_loss = self._anchor(compressed, mse_ref)
            elif not self._warned_no_adjacency:
                warnings.warn(
                    "CompositeLoss: graph term is active but adjacency=None — "
                    "the graph term contributes a constant zero. Pass the "
                    "prior adjacency through the trainer.",
                    RuntimeWarning,
                )
                self._warned_no_adjacency = True

            losses.append(self.graph_weight * graph_loss)
            components["graph"] = graph_loss.item()
            components["graph_raw"] = graph_raw.item()
            self.loss_history["graph"].append(graph_loss.item())
            self.loss_history["graph_raw"].append(graph_raw.item())

        # ── 5. Aggregate ──────────────────────────────────────────────────────
        assert len(losses) == self.n_tasks, (
            f"CompositeLoss built for n_tasks={self.n_tasks} but produced "
            f"{len(losses)} terms in mode '{self.ablation_mode}'."
        )
        if self.uncertainty is not None:
            total_loss, uw_log = self.uncertainty(losses)
            components.update(uw_log)
        else:
            total_loss = torch.stack(losses).sum()

        components["total"] = total_loss.item()
        return (total_loss, components) if return_components else total_loss

    def _graph_regularization(self, x_pred: torch.Tensor,
                              adjacency: torch.Tensor) -> torch.Tensor:
        """
        Graph smoothness regulariser: penalises large differences between
        sensors that the prior graph says are coupled.

            L_graph = sum_ij A_ij |x_i - x_j|  /  sum_ij A_ij

        x_pred    : (batch, d)
        adjacency : (d, d), already on x_pred's device
        returns   : scalar

        Normalising by the edge count keeps the magnitude comparable across
        graphs of different densities; a raw .sum() grows with graph size and
        would make graph_weight dataset-specific.

        Self-loops carry zero difference by construction, so counting them in
        the denominator only dilutes the term — they are dropped when
        graph_ignore_self_loops is True.

        The gradient flows through x_pred, so this term does influence
        training whenever it is active (unlike the physics term with
        physics_on_prediction=False).
        """
        adj = adjacency
        if self.graph_ignore_self_loops:
            adj = adj * (1.0 - torch.eye(adj.shape[0], device=adj.device,
                                         dtype=adj.dtype))

        diff = x_pred.unsqueeze(2) - x_pred.unsqueeze(1)      # (B, d, d)
        diff_norm = diff.abs().mean(dim=0)                    # (d, d)

        weighted = (adj * diff_norm).sum()
        n_edges = adj.sum().clamp(min=1.0)
        return weighted / n_edges

    def get_loss_history(self):
        return self.loss_history

    def reset_history(self):
        self.loss_history = {
            "mse": [], "physics": [], "physics_raw": [],
            "temporal": [], "temporal_raw": [],
            "graph": [], "graph_raw": [],
        }
        self._warned_no_adjacency = False
        self._warned_no_edges = False


# These are the unchanged uncertainty classes — kept here for self-contained use.
class UncertaintyWeighting(nn.Module):
    def __init__(self, n_tasks: int):
        super().__init__()
        self.n_tasks = n_tasks

    def forward(self, losses):
        raise NotImplementedError


class KendallGalUncertainty(UncertaintyWeighting):
    def __init__(self, n_tasks: int, init_log_sigma: float = 0.0):
        super().__init__(n_tasks)
        self.log_sigma = nn.Parameter(
            torch.full((n_tasks,), init_log_sigma, dtype=torch.float32)
        )

    def forward(self, losses):
        losses = torch.stack(losses, dim=0)
        if losses.dim() > 1:
            losses = losses.mean(dim=-1)
        sigma = torch.exp(self.log_sigma)
        weighted = losses / (2.0 * sigma ** 2) + self.log_sigma
        total_loss = weighted.sum()
        log_dict = {f"sigma_{i}": sigma[i].item() for i in range(self.n_tasks)}
        return total_loss, log_dict


class GradNormWeighting(UncertaintyWeighting):
    def __init__(self, n_tasks: int, alpha: float = 1.5):
        super().__init__(n_tasks)
        self.weights = nn.Parameter(torch.ones(n_tasks, dtype=torch.float32))
        self.alpha = alpha
        self.initial_losses = None

    def forward(self, losses, grad_norms=None):
        losses = torch.stack(losses, dim=0)
        if losses.dim() > 1:
            losses = losses.mean(dim=-1)
        if self.initial_losses is None:
            self.initial_losses = losses.detach()
        normalized = losses / (self.initial_losses + 1e-8)
        weighted = (self.weights * normalized).sum()
        log_dict = {f"w_{i}": self.weights[i].item() for i in range(self.n_tasks)}
        return weighted, log_dict


class DynamicWeightAveraging(UncertaintyWeighting):
    def __init__(self, n_tasks: int, temperature: float = 2.0):
        super().__init__(n_tasks)
        self.temperature = temperature
        self.prev_losses = None

    def forward(self, losses):
        losses = torch.stack(losses, dim=0)
        if losses.dim() > 1:
            losses = losses.mean(dim=-1)
        if self.prev_losses is None:
            self.prev_losses = losses.detach()
            weights = torch.ones(self.n_tasks) / self.n_tasks
        else:
            improvement = (self.prev_losses - losses).detach()
            weights = torch.softmax(improvement / self.temperature, dim=0)
            self.prev_losses = losses.detach()
        weighted = (weights * losses).sum()
        log_dict = {f"w_{i}": weights[i].item() for i in range(self.n_tasks)}
        return weighted, log_dict