from pydantic import BaseModel
from typing import Optional, Dict, List


class MWResult(BaseModel):
    U: Optional[float] = None
    p_value: Optional[float] = None


class SessionOut(BaseModel):
    id: int
    patient_id: int
    test_type: str
    status: str
    quality_score: Optional[float] = None
    error_message: Optional[str] = None

    features: Dict[str, float] = {}
    stats: Dict[str, MWResult] = {}

    conclusion: Optional[str] = None
    risk_score: Optional[float] = None

    report_path: Optional[str] = None
    csv_path: Optional[str] = None


class LOOCVResultOut(BaseModel):
    classifier: str
    n_samples: int
    n_bp: int
    n_kg: int
    accuracy: float
    f1: float
    roc_auc: Optional[float] = None
    pr_auc: Optional[float] = None
    confusion: List[List[int]]
    feature_importance: Dict[str, float] = {}


class LOOCVResponse(BaseModel):
    test_type: str
    results: List[LOOCVResultOut]
