"""
Janus 电路转换器

提供与 Qiskit 等其他框架的互转功能
"""
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from qiskit import QuantumCircuit as QiskitCircuit

from .circuit import Circuit
from .gate import Gate
from .library.standard_gates import (
    HGate, XGate, YGate, ZGate, SGate, TGate,
    RXGate, RYGate, RZGate, UGate,
    CXGate, CZGate, CRZGate, SwapGate
)


# 门名称到类的映射
GATE_MAP = {
    'h': HGate,
    'x': XGate,
    'y': YGate,
    'z': ZGate,
    's': SGate,
    't': TGate,
    'rx': RXGate,
    'ry': RYGate,
    'rz': RZGate,
    'u': UGate,
    'u3': UGate,  # u3 等价于 u
    'cx': CXGate,
    'cz': CZGate,
    'crz': CRZGate,
    'swap': SwapGate,
}


def to_qiskit(circuit: Circuit, barrier: bool = False) -> 'QiskitCircuit':
    """
    将 Janus Circuit 转换为 Qiskit QuantumCircuit
    
    Args:
        circuit: Janus 电路
        barrier: 是否在每层之间添加 barrier
    
    Returns:
        Qiskit QuantumCircuit
    """
    try:
        from qiskit import QuantumCircuit as QiskitCircuit
    except ImportError:
        raise ImportError("Qiskit is required for this conversion. Install with: pip install qiskit")
    
    qc = QiskitCircuit(circuit.n_qubits)
    
    if barrier:
        # 按层添加
        for layer in circuit.layers:
            for inst in layer:
                _add_gate_to_qiskit(qc, inst.name, inst.qubits, inst.params)
            qc.barrier()
    else:
        # 按顺序添加
        for inst in circuit.instructions:
            _add_gate_to_qiskit(qc, inst.name, inst.qubits, inst.params)
    
    return qc


def _add_gate_to_qiskit(qc, name: str, qubits: list, params: list):
    """向 Qiskit 电路添加门"""
    name = name.lower()
    
    if name in ('rx', 'ry', 'rz'):
        getattr(qc, name)(float(params[0]), qubits[0])
    elif name in ('cx', 'cz', 'swap'):
        getattr(qc, name)(qubits[0], qubits[1])
    elif name in ('h', 'x', 'y', 'z', 's', 't'):
        getattr(qc, name)(qubits[0])
    elif name in ('u', 'u3'):
        qc.u(float(params[0]), float(params[1]), float(params[2]), qubits[0])
    elif name == 'crz':
        qc.crz(float(params[0]), qubits[0], qubits[1])
    elif name == 'barrier':
        pass  # 跳过 barrier
    else:
        print(f"Warning: Unknown gate '{name}', skipping")


def from_qiskit(qiskit_circuit: 'QiskitCircuit') -> Circuit:
    """
    将 Qiskit QuantumCircuit 转换为 Janus Circuit
    
    Args:
        qiskit_circuit: Qiskit 量子电路
    
    Returns:
        Janus Circuit
    """
    circuit = Circuit(qiskit_circuit.num_qubits)
    
    for instruction in qiskit_circuit.data:
        op = instruction.operation
        name = op.name.lower()
        qubits = [q._index for q in instruction.qubits]
        params = [float(p) if not isinstance(p, float) else p for p in op.params]
        
        # 跳过 barrier 和 measure
        if name in ('barrier', 'measure'):
            continue
        
        # 创建对应的门
        gate = _create_gate(name, params)
        if gate is not None:
            circuit.append(gate, qubits)
    
    return circuit


def _create_gate(name: str, params: list) -> Optional[Gate]:
    """根据名称和参数创建门"""
    name = name.lower()
    
    if name == 'u3':
        name = 'u'
    
    if name not in GATE_MAP:
        print(f"Warning: Unknown gate '{name}', skipping")
        return None
    
    gate_class = GATE_MAP[name]
    
    # 根据门类型创建实例
    if name in ('h', 'x', 'y', 'z', 's', 't', 'cx', 'cz', 'swap'):
        return gate_class()
    elif name in ('rx', 'ry', 'rz', 'crz'):
        return gate_class(params[0])
    elif name == 'u':
        return gate_class(params[0], params[1], params[2])
    
    return None
