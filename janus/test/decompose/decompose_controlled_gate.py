# 测试受控门分解函数
import sys
import os

# 添加项目根目录到sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import numpy as np

# 从 Janus 包导入所需的类
from circuit.circuit import Circuit
from circuit.library.standard_gates import XGate, YGate, ZGate, RXGate, RYGate, RZGate

# 导入分解函数
from janus.decompose.decompose_controlled_gate import decompose_controlled_gate

def print_test_result(func_name, success, details=""):
    """打印测试结果"""
    status = "✓ SUCCESS" if success else "✗ FAILED"
    print(f"{func_name}: {status} {details}")

def test_decompose_controlled_gate():
    """测试受控门分解函数"""
    print("\n=== 测试受控门分解函数 (decompose_controlled_gate) ===")
    
    # 测试不同的基础门
    gates = [
        (XGate(), "XGate"),
        (YGate(), "YGate"),
        (ZGate(), "ZGate"),
        (RXGate(0.5), "RXGate"),
        (RYGate(0.5), "RYGate"),
        (RZGate(0.5), "RZGate")
    ]
    
    # 测试 2 控制门分解
    for gate, gate_name in gates:
        try:
            decomposed_gate = decompose_controlled_gate(gate, num_ctrl_qubits=2)
            print_test_result(f"2控制{gate_name}分解(hp24)", True, f"深度: {decomposed_gate.depth}, 门数: {decomposed_gate.n_gates}")
        except Exception as e:
            print_test_result(f"2控制{gate_name}分解", False, str(e))
    
    # 测试 3 控制 X 门的分解
    try:
        decomposed_3cx = decompose_controlled_gate(XGate(), num_ctrl_qubits=3)
        print_test_result("3控制X门分解(hp24)", True, f"深度: {decomposed_3cx.depth}, 门数: {decomposed_3cx.n_gates}")
    except Exception as e:
        print_test_result("3控制X门分解(hp24)", False, str(e))

if __name__ == "__main__":
    test_decompose_controlled_gate()