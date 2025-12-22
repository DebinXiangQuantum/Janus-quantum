# 测试KAK分解函数
import sys
import os
import numpy as np

# 添加项目根目录到sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

# 导入Janus库的组件
from circuit.library.standard_gates import (
    HGate, CXGate, CZGate, SwapGate
)
from circuit.circuit import Circuit

# 导入分解函数
from janus.decompose.decompose_kak import decompose_kak

def random_unitary(dim):
    """生成随机单位矩阵"""
    from scipy.stats import unitary_group
    return unitary_group.rvs(dim)

def print_test_result(func_name, success, details=""):
    """打印测试结果"""
    status = "✓ SUCCESS" if success else "✗ FAILED"
    print(f"{func_name}: {status} {details}")

def test_decompose_kak():
    """测试任意数量量子比特KAK分解函数"""
    print("\n=== 测试任意数量量子比特KAK分解函数 (decompose_kak) ===")
    
    # 测试1量子比特门分解
    h_gate = HGate()
    try:
        decomposed_h = decompose_kak(h_gate, euler_basis='ZXZ', simplify=True)
        print_test_result("1量子比特H门KAK分解", True, f"深度: {decomposed_h.depth}, 门数: {len(decomposed_h._instructions)}")
    except Exception as e:
        print_test_result("1量子比特H门KAK分解", False, str(e))
    
    # 测试2量子比特门分解
    two_qubit_gates = [
        (CXGate(), "CXGate"),
        (CZGate(), "CZGate"),
        (SwapGate(), "SwapGate")
    ]
    
    for gate, gate_name in two_qubit_gates:
        try:
            decomposed_gate = decompose_kak(gate, euler_basis='ZXZ', simplify=True)
            print_test_result(f"2量子比特{gate_name}KAK分解", True, f"深度: {decomposed_gate.depth}, 门数: {len(decomposed_gate._instructions)}")
        except Exception as e:
            print_test_result(f"2量子比特{gate_name}KAK分解", False, str(e))
    
    # 测试3量子比特随机幺正矩阵分解
    try:
        random_3q = random_unitary(8)
        decomposed_3q = decompose_kak(random_3q, euler_basis='ZXZ', simplify=True)
        print_test_result("3量子比特随机幺正矩阵KAK分解", True, f"深度: {decomposed_3q.depth}, 门数: {len(decomposed_3q._instructions)}")
    except Exception as e:
        print_test_result("3量子比特随机幺正矩阵KAK分解", False, str(e))

if __name__ == "__main__":
    test_decompose_kak()