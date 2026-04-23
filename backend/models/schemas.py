"""Pydantic schemas for API request/response models."""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class RiskLevel(str, Enum):
    CRITICAL   = "CRITICAL"
    HIGH       = "HIGH"
    MEDIUM     = "MEDIUM"
    LOW        = "LOW"
    NEGLIGIBLE = "NEGLIGIBLE"


class AttackerMetrics(BaseModel):
    accuracy:           float
    balanced_accuracy:  float
    auc_roc:            float
    leakage_score:      float
    mutual_information: float


class AttributeAuditResult(BaseModel):
    n_classes:           int
    labels:              List[str]
    logistic_regression: AttackerMetrics
    mlp:                 AttackerMetrics
    best_attacker_auc:   float
    best_leakage_score:  float
    risk_level:          RiskLevel


class AuditSummary(BaseModel):
    mean_leakage_score:    float
    max_attacker_auc:      float
    overall_risk_level:    RiskLevel
    n_attributes_audited:  int


class AuditResult(BaseModel):
    gender:      Optional[AttributeAuditResult] = None
    age_group:   Optional[AttributeAuditResult] = None
    ethnicity:   Optional[AttributeAuditResult] = None
    summary:     Optional[AuditSummary]         = None
    tag:         str = "baseline"
    status:      str = "completed"


class ComparisonRow(BaseModel):
    attribute:           str
    baseline_auc:        float
    adv_auc:             float
    noise_auc:           float
    adv_reduction:       float
    noise_reduction:     float
    adv_reduction_pct:   float
    noise_reduction_pct: float
    baseline_risk:       RiskLevel
    adv_risk:            RiskLevel
    noise_risk:          RiskLevel


class SweepPoint(BaseModel):
    param_value:          float
    mean_leakage_score:   float
    identity_utility:     float


class MitigationResult(BaseModel):
    comparison:    List[ComparisonRow]  = []
    adv_sweep:     List[Dict[str, Any]] = []
    noise_sweep:   List[Dict[str, Any]] = []
    status:        str = "completed"


class GroupLeakage(BaseModel):
    mean_leakage: float
    max_leakage:  float


class FairnessReport(BaseModel):
    per_group_leakage:         Dict[str, GroupLeakage] = {}
    max_demographic_disparity: float = 0.0
    fairness_gap:              float = 0.0
    n_groups_analyzed:         int   = 0


class ModelCard(BaseModel):
    model_card_version:  str
    framework:           str
    generated_at:        str
    overall_risk_level:  str
    mean_leakage_score:  float
    recommendations:     List[str]
    attribute_narratives: Dict[str, str]  = {}
    visualizations:      Dict[str, str]   = {}


class PipelineStatus(BaseModel):
    status:    str   # "idle" | "running" | "completed" | "error"
    progress:  float = 0.0
    message:   str   = ""
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error:     Optional[str]   = None


class RunPipelineRequest(BaseModel):
    dataset:    str = "synthetic"
    run_sweep:  bool = True
    tag:        str = "user_run"
