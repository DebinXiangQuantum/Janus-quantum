"""
Janus - 轻量级量子电路框架

Janus 是一个用于量子电路构建、模拟和编译的 Python 框架。

Example:
    >>> from janus.circuit import Circuit
    >>> from janus.simulator import StatevectorSimulator
    >>>
    >>> qc = Circuit(2)
    >>> qc.h(0)
    >>> qc.cx(0, 1)
    >>>
    >>> sim = StatevectorSimulator()
    >>> result = sim.run(qc, shots=1000)
    >>> print(result.counts)
"""

__version__ = "1.0.0"
__author__ = "Janus Team"

from janus import circuit
from janus import simulator
from janus import compiler
from janus import decompose
from janus import optimize
from janus import encode

__all__ = [
    "circuit",
    "simulator",
    "compiler",
    "decompose",
    "optimize",
    "encode",
    "__version__",
]
