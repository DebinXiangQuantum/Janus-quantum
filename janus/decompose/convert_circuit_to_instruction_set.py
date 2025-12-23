from __future__ import annotations
from typing import Optional, List, Union
import numpy as np
from circuit import Circuit, Gate
from circuit.converters import GATE_MAP
from circuit.dag import circuit_to_dag
from circuit.library.standard_gates import (
    HGate, XGate, YGate, ZGate, SGate, TGate,
    RXGate, RYGate, RZGate, UGate,
    CXGate, CZGate, CRZGate, SwapGate
)
from .exceptions import ParameterError, GateNotSupportedError


def convert_circuit_to_instruction_set(
    circuit: Circuit,
    instruction_set: List[str],
    coupling_map: Optional[Union[List[List[int]]]] = None,
    optimization_level: Optional[int] = 1,
    seed_transpiler: Optional[int] = None,
    use_dag: bool = False,
) -> Circuit:
    """
    将电路转换为指定的指令集
    
    Args:
        circuit: Janus电路对象
        instruction_set: 目标指令集（门名称列表）
        coupling_map: 耦合映射（未实现）
        optimization_level: 优化级别（未实现）
        seed_transpiler: 随机种子（未实现）
        use_dag: 是否返回DAGCircuit
        
    Returns:
        转换后的电路
    """
    if not isinstance(circuit, Circuit):
        raise ParameterError("Input must be a valid Janus Circuit object.")
    
    if not isinstance(instruction_set, list) or not instruction_set:
        raise ParameterError("instruction_set must be a non-empty list of strings.")
    
    if not all(isinstance(gate, str) for gate in instruction_set):
        raise ParameterError("All elements in instruction_set must be strings.")
    
    # 检查指令集中的门是否都支持
    for gate_name in instruction_set:
        if gate_name not in GATE_MAP:
            raise GateNotSupportedError(f"Unsupported gate type: {gate_name}")
    
    # 创建新电路
    converted_circuit = Circuit(n_qubits=circuit.n_qubits)
    
    # 转换每个门
    for inst in circuit.instructions:
        gate_name = inst.name.lower()
        qubits = inst.qubits
        params = inst.params
        
        # 如果门已经在目标指令集中，直接添加
        if gate_name in instruction_set:
            gate_cls = GATE_MAP[gate_name]
            if gate_cls == UGate:
                # U门需要三个参数：theta, phi, lam
                if len(params) == 0:
                    # 默认参数
                    converted_circuit.append(gate_cls(0, 0, 0), qubits)
                elif len(params) == 1:
                    # 只有theta
                    converted_circuit.append(gate_cls(params[0], 0, 0), qubits)
                elif len(params) == 2:
                    # theta和phi
                    converted_circuit.append(gate_cls(params[0], params[1], 0), qubits)
                else:
                    # 完整参数
                    converted_circuit.append(gate_cls(params[0], params[1], params[2]), qubits)
            elif gate_cls in [RXGate, RYGate, RZGate]:
                # 旋转门需要一个角度参数
                if params:
                    converted_circuit.append(gate_cls(params[0]), qubits)
                else:
                    converted_circuit.append(gate_cls(0), qubits)
            else:
                # 其他门直接添加
                converted_circuit.append(gate_cls(params), qubits)
        else:
            # 分解实现
            if gate_name in ['h', 'x', 'y', 'z', 's', 't', 'rx', 'ry', 'rz', 'u']:
                # 单比特门
                if 'u' in instruction_set:
                    # 转换为U门
                    if gate_name == 'h':
                        # H门: U(π/2, 0, π)
                        converted_circuit.append(UGate(np.pi/2, 0, np.pi), qubits)
                    elif gate_name == 'x':
                        # X门: U(π, 0, π)
                        converted_circuit.append(UGate(np.pi, 0, np.pi), qubits)
                    elif gate_name == 'y':
                        # Y门: U(π, π/2, π/2)
                        converted_circuit.append(UGate(np.pi, np.pi/2, np.pi/2), qubits)
                    elif gate_name == 'z':
                        # Z门: U(0, 0, π)
                        converted_circuit.append(UGate(0, 0, np.pi), qubits)
                    elif gate_name == 's':
                        # S门: U(0, 0, π/2)
                        converted_circuit.append(UGate(0, 0, np.pi/2), qubits)
                    elif gate_name == 't':
                        # T门: U(0, 0, π/4)
                        converted_circuit.append(UGate(0, 0, np.pi/4), qubits)
                    elif gate_name in ['rx', 'ry', 'rz']:
                        # 旋转门转换为U门
                        if params:
                            theta = params[0]
                            if gate_name == 'rx':
                                converted_circuit.append(UGate(theta, -np.pi/2, np.pi/2), qubits)
                            elif gate_name == 'ry':
                                converted_circuit.append(UGate(theta, 0, 0), qubits)
                            else:  # rz
                                converted_circuit.append(UGate(0, 0, theta), qubits)
                        else:
                            converted_circuit.append(UGate(0, 0, 0), qubits)
                    elif gate_name == 'u':
                        # U门需要三个参数
                        if len(params) >= 3:
                            converted_circuit.append(UGate(params[0], params[1], params[2]), qubits)
                        elif len(params) == 2:
                            converted_circuit.append(UGate(params[0], params[1], 0), qubits)
                        elif len(params) == 1:
                            converted_circuit.append(UGate(params[0], 0, 0), qubits)
                        else:
                            converted_circuit.append(UGate(0, 0, 0), qubits)
                elif all(g in instruction_set for g in ['rx', 'ry', 'rz']):
                    # 转换为RX/RY/RZ组合
                    converted_circuit.append(RXGate(np.pi/2), qubits)
                    converted_circuit.append(RYGate(np.pi/4), qubits)
                    converted_circuit.append(RZGate(np.pi/8), qubits)
                else:
                    raise ValueError(f"Cannot decompose {gate_name} into target instruction set")
            elif gate_name in ['cx', 'cz', 'crz', 'swap']:
                # 两比特门
                if 'cx' in instruction_set:
                    # 转换为CX门
                    converted_circuit.append(CXGate(), qubits)
                else:
                    raise ValueError(f"Cannot decompose {gate_name} into target instruction set")
    
    if use_dag:
        return circuit_to_dag(converted_circuit)
    return converted_circuit