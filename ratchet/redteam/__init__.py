"""
RATCHET Red Team Module

Adversarial testing framework implementing attacks RT-01 through RT-05
from the Formal Methods Review.

Attacks:
- RT-01: Emergent Deception (exploits L-01 limitation)
- RT-02: Slow Capture (gradual node compromise)
- RT-03: Mimicry (statistical camouflage)
- RT-04: Flooding (noise injection)
- RT-05: Diverse Sybils (bypasses MI gate)
"""

from .attacks import (
    Attack,
    AttackResult,
    RT01_EmergentDeception,
    RT02_SlowCapture,
    RT03_Mimicry,
    RT04_Flooding,
    RT05_DiverseSybils,
)

from .harness import (
    RedTeamHarness,
    AttackSuite,
)

__all__ = [
    'Attack',
    'AttackResult',
    'RT01_EmergentDeception',
    'RT02_SlowCapture',
    'RT03_Mimicry',
    'RT04_Flooding',
    'RT05_DiverseSybils',
    'RedTeamHarness',
    'AttackSuite',
]
