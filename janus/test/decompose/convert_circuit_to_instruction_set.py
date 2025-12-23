# 测试电路到指令集转换函数
import sys
import os

# 添加项目根目录到sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import numpy as np
from circuit import Circuit

# 导入分解函数
from janus.decompose.convert_circuit_to_instruction_set import convert_circuit_to_instruction_set


def print_test_result(func_name, success, details=""):
    """打印测试结果"""
    status = "✓ SUCCESS" if success else "✗ FAILED"
    print(f"{func_name}: {status} {details}")


def test_convert_circuit_to_instruction_set():
    """测试电路到指令集转换函数"""
    print("\n=== 测试电路到指令集转换函数 (convert_circuit_to_instruction_set) ===")
    
    # 创建测试电路
    def create_test_circuit(num_qubits=2):
        qc = Circuit(num_qubits)
        qc.h(0)
        for i in range(num_qubits - 1):
            qc.cx(i, i + 1)
        # 单个量子比特门需要逐个应用
        for i in range(num_qubits):
            qc.z(i)
        qc.x(0)
        qc.y(num_qubits - 1)
        return qc
    
    # 测试2量子比特电路
    qc2 = create_test_circuit(2)
    
    # 测试不同的指令集
    instruction_sets = [
        (['u', 'cx'], "u + cx"),
        (['u', 'cx', 'rz'], "u + cx + rz")
    ]
    
    for inst_set, inst_set_name in instruction_sets:
        try:
            converted_qc = convert_circuit_to_instruction_set(qc2, inst_set)
            # 获取门数
            gate_counts = {}
            for inst in converted_qc.instructions:
                gate_name = inst.name.lower()
                gate_counts[gate_name] = gate_counts.get(gate_name, 0) + 1
            
            print_test_result(f"电路转换为{inst_set_name}指令集", True, f"深度: {converted_qc.depth}, 门数: {gate_counts}")
        except Exception as e:
            print_test_result(f"电路转换为{inst_set_name}指令集", False, str(e))
    
    # 测试带耦合映射的转换
    try:
        qc3 = create_test_circuit(3)
        coupling_map = [[0, 1], [1, 2]]  # 线性链
        converted_qc3 = convert_circuit_to_instruction_set(
            qc3, ['u', 'cx']
            
        )
        gate_counts = {}
        for inst in converted_qc3.instructions:
            gate_name = inst.name.lower()
            gate_counts[gate_name] = gate_counts.get(gate_name, 0) + 1
        print_test_result("3量子比特电路转换", True, f"深度: {converted_qc3.depth}, 门数: {gate_counts}")
    except Exception as e:
        print_test_result("3量子比特电路转换", False, str(e))


if __name__ == "__main__":
    test_convert_circuit_to_instruction_set()