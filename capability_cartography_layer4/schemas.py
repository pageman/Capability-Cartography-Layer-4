from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum

# Mocking torch for environments without it
class MockTensor:
    def __init__(self, data=None):
        self.shape = (1, 1)
        self.device = "cpu"
    def mean(self): return MockTensor()
    def item(self): return 0.0
    def argmax(self, dim): return MockTensor()
    def float(self): return MockTensor()
    def squeeze(self): return MockTensor()

class MockTorch:
    float = "float"
    long = "long"
    Tensor = MockTensor
    def tensor(self, data, dtype=None): return MockTensor(data)
    def randn(self, *args): return MockTensor()
    def diagonal(self, x): return MockTensor()
    def matmul(self, a, b): return MockTensor()
    def softmax(self, x, dim): return MockTensor()
    def randint(self, *args): return MockTensor()

class TrajectoryType(Enum):
    POWER_LAW = "power-law"
    EMERGENT = "emergent"
    STEP_FUNCTION = "step-function"
    THEORETICAL = "theoretical"
    POWER_LAW_WITH_BIAS = "power-law-with-bias"
    HYBRID = "hybrid"

class VerdictType(Enum):
    CONFIRMED = "CONFIRMED"
    CONDITIONAL = "CONDITIONAL"
    UNCONFIRMED = "UNCONFIRMED"
    UNDETECTED = "UNDETECTED"

class CircuitType(Enum):
    INDUCTION = "induction_head"
    FOURIER = "fourier_based"
    SPARSE_AUTOENCODER = "sae_feature"
    ACDC_SUBGRAPH = "acdc_subgraph"
    UNKNOWN = "unknown"

@dataclass
class RegimeProfile:
    """Pre-training regime parameters from task descriptors."""
    m: int  # Number of environments/instruments
    r: float  # Samples per environment (n/m)
    d: int  # Dimensionality
    s_star: int  # Sparsity
    splitup_needed: bool
    regime_label: str

@dataclass
class TrajectoryForecast:
    """Forecast made BEFORE training begins."""
    type: TrajectoryType
    predicted_exponent: Optional[float] = None
    emergence_threshold: Optional[float] = None
    scale_transfer_invariant: bool = False
    confidence: float = 0.0
    pathology_risk: Optional[str] = None 

@dataclass
class QuantumAnalogyAnnotation:
    """Interpretive, non-verified analogy between a mechanism and a quantum motif."""
    analogy_name: str
    register_ids: List[int]
    classical_component: str
    confidence: float
    rationale: str

@dataclass
class MechanismOperation:
    """An instantiated mechanistic operation bound to a concrete model component."""
    operation_id: str
    component: str
    role: str
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    metadata: Dict[str, str] = field(default_factory=dict)

@dataclass
class MechanismMoment:
    """A stage of non-conflicting mechanistic operations, inspired by Cirq moments."""
    stage: int
    operations: List[MechanismOperation] = field(default_factory=list)
    label: Optional[str] = None

@dataclass
class MechanismCircuit:
    """Structured mechanism IR for staged interpretability analysis."""
    moments: List[MechanismMoment] = field(default_factory=list)
    interpretive_quantum_analogies: List[QuantumAnalogyAnnotation] = field(default_factory=list)
    analogy_summary: Optional[str] = None

@dataclass
class CircuitDefinition:
    """Mechanistic description of the capability internal implementation."""
    type: CircuitType
    components: List[str]
    mechanism_description: str
    quantum_connection_potential: bool = False
    fourier_signature: Optional[float] = None
    mechanism_circuit: Optional[MechanismCircuit] = None

@dataclass
class EmergenceRecord:
    """Tracking of circuit formation during training checkpoints."""
    grokking_detected: bool
    grokking_step: Optional[int]
    accuracy_history: Dict[int, float]
    circuit_stability_score: float

@dataclass
class CausalVerification:
    """Intervention-based verification results."""
    ablation_accuracy_drop: float
    injection_accuracy_gain: Optional[float]
    severing_impact: str
    abstraction_faithful: bool

@dataclass
class CCL4Record:
    """Unified record for a capability: Predictive + Mechanistic."""
    capability_id: str
    paper_id: str
    regime: RegimeProfile
    forecast: TrajectoryForecast
    circuit: Optional[CircuitDefinition] = None
    emergence: Optional[EmergenceRecord] = None
    verification: Optional[CausalVerification] = None
    verdict: VerdictType = VerdictType.UNDETECTED
    verdict_confidence: float = 0.0
    provenance: Dict[str, str] = field(default_factory=dict)
