# This code is part of Qiskit.
#
# (C) Copyright IBM 2023, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Controlled gate decomposition interface.
"""
from __future__ import annotations

from qiskit.circuit import Gate, ControlledGate, QuantumCircuit, QuantumRegister
from qiskit.circuit.exceptions import CircuitError

# 支持的受控门分解方法
SUPPORTED_METHODS = {
    "default": "使用默认分解方法",
    "noancilla": "无辅助量子比特分解",
    "gray": "格雷码分解方法",
    "b95": "Barenco 1995分解方法",
    "i15": "Iten 2015分解方法",
    "m15": "Maslov 2015分解方法",
    "kg24": "Khattar和Gidney 2024分解方法",
    "hp24": "高效无辅助量子比特方法（线性门数）"
}

def decompose_controlled_gate(
    gate: Gate | ControlledGate,
    num_ctrl_qubits: int | None = None,
    method: str = "default",
    num_ancilla_qubits: int = 0,
    ancilla_type: str = "clean"
) -> QuantumCircuit:
    """
    将受控门分解为基础门序列的统一接口函数。
    
    Args:
        gate: 输入的门或受控门。如果是普通门，将首先添加控制量子比特。
        num_ctrl_qubits: 控制量子比特数量。如果gate已经是受控门，此参数将被忽略。
        method: 分解方法，可选值包括：
            - "default": 使用默认分解方法
            - "noancilla": 无辅助量子比特分解
            - "gray": 格雷码分解方法
            - "b95": Barenco 1995分解方法
            - "i15": Iten 2015分解方法
            - "m15": Maslov 2015分解方法
            - "kg24": Khattar和Gidney 2024分解方法
            - "hp24": 高效无辅助量子比特方法（线性门数）
        num_ancilla_qubits: 辅助量子比特数量。
        ancilla_type: 辅助量子比特类型，可选"clean"或"dirty"。
    
    Returns:
        QuantumCircuit: 包含分解后门序列的量子电路。
    
    Raises:
        CircuitError: 如果输入的门不支持分解或参数无效。
    """
    if method not in SUPPORTED_METHODS:
        raise CircuitError(f"不支持的分解方法: {method}。支持的方法包括: {', '.join(SUPPORTED_METHODS.keys())}")
    
    # 处理输入的门
    controlled_gate = _get_controlled_gate(gate, num_ctrl_qubits)
    
    # 根据门类型和方法选择分解策略
    if controlled_gate.base_gate.name == "x":
        # 多控制X门（Toffoli门）
        from qiskit.synthesis.multi_controlled import decompose_multi_control_toffoli
        return decompose_multi_control_toffoli(
            num_ctrl_qubits=controlled_gate.num_ctrl_qubits,
            method=method,
            num_ancilla_qubits=num_ancilla_qubits,
            ancilla_type=ancilla_type
        )
    elif controlled_gate.base_gate.name in ["rx", "ry", "rz", "p"]:
        # 多控制旋转门
        return _decompose_controlled_rotation_gate(controlled_gate, method)
    elif controlled_gate.num_qubits <= 3:
        # 3量子比特或更少的受控门
        return _decompose_small_controlled_gate(controlled_gate, method)
    else:
        # 通用受控门分解
        return _decompose_general_controlled_gate(controlled_gate, method)

def _get_controlled_gate(gate: Gate | ControlledGate, num_ctrl_qubits: int | None = None) -> ControlledGate:
    """
    获取受控门。如果输入是普通门，添加控制量子比特。
    
    Args:
        gate: 输入的门或受控门。
        num_ctrl_qubits: 控制量子比特数量。
    
    Returns:
        ControlledGate: 受控门实例。
    """
    from qiskit.circuit import ControlledGate
    
    if isinstance(gate, ControlledGate):
        return gate
    
    if num_ctrl_qubits is None:
        num_ctrl_qubits = 1
    
    # 使用add_control函数添加控制量子比特
    from qiskit.circuit._add_control import add_control
    return add_control(gate, num_ctrl_qubits=num_ctrl_qubits, label=None, ctrl_state=None)

def _decompose_controlled_rotation_gate(controlled_gate: ControlledGate, method: str) -> QuantumCircuit:
    """
    分解多控制旋转门。
    
    Args:
        controlled_gate: 多控制旋转门。
        method: 分解方法。
    
    Returns:
        QuantumCircuit: 分解后的量子电路。
    """
    num_ctrl_qubits = controlled_gate.num_ctrl_qubits
    target_qubit = QuantumRegister(1, name="target")
    ctrl_qubits = QuantumRegister(num_ctrl_qubits, name="control")
    circuit = QuantumCircuit(ctrl_qubits, target_qubit, name=controlled_gate.name)
    
    # 根据旋转轴选择合适的分解方法
    if controlled_gate.base_gate.name == "rx":
        circuit.mcrx(controlled_gate.params[0], ctrl_qubits, target_qubit)
    elif controlled_gate.base_gate.name == "ry":
        circuit.mcry(controlled_gate.params[0], ctrl_qubits, target_qubit)
    elif controlled_gate.base_gate.name == "rz":
        circuit.mcrz(controlled_gate.params[0], ctrl_qubits, target_qubit)
    elif controlled_gate.base_gate.name == "p":
        from qiskit.circuit.library import MCPhaseGate
        circuit.append(MCPhaseGate(controlled_gate.params[0], num_ctrl_qubits), ctrl_qubits[:] + target_qubit[:])
    
    return circuit

def _decompose_small_controlled_gate(controlled_gate: ControlledGate, method: str) -> QuantumCircuit:
    """
    分解3量子比特或更少的受控门。
    
    Args:
        controlled_gate: 受控门。
        method: 分解方法。
    
    Returns:
        QuantumCircuit: 分解后的量子电路。
    """
    # 创建一个包含受控门的电路
    qr = QuantumRegister(controlled_gate.num_qubits)
    circuit = QuantumCircuit(qr, name=controlled_gate.name)
    circuit.append(controlled_gate, qr)
    
    # 使用BasisTranslator进行分解
    from qiskit.transpiler import PassManager
    from qiskit.transpiler.passes import Unroll3qOrMore, BasisTranslator
    from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary as sel
    
    pm = PassManager([
        Unroll3qOrMore(),
        BasisTranslator(sel, target_basis=["u", "p", "cx"])
    ])
    
    return pm.run(circuit)

def _decompose_general_controlled_gate(controlled_gate: ControlledGate, method: str) -> QuantumCircuit:
    """
    分解通用受控门。
    
    Args:
        controlled_gate: 受控门。
        method: 分解方法。
    
    Returns:
        QuantumCircuit: 分解后的量子电路。
    """
    # 创建一个包含受控门的电路
    qr = QuantumRegister(controlled_gate.num_qubits)
    circuit = QuantumCircuit(qr, name=controlled_gate.name)
    circuit.append(controlled_gate, qr)
    
    # 使用BasisTranslator进行分解
    from qiskit.transpiler import PassManager
    from qiskit.transpiler.passes import Unroll3qOrMore, BasisTranslator
    from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary as sel
    
    pm = PassManager([
        Unroll3qOrMore(),
        BasisTranslator(sel, target_basis=["u", "p", "cx"])
    ])
    
    return pm.run(circuit)
