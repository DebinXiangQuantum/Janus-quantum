# 测试单量子比特门分解函数
import sys
import os

# 添加项目根目录到sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
import numpy as np
from circuit.circuit import Circuit
from circuit.library.standard_gates import HGate, XGate

# 导入分解函数
from janus.decompose.decompose_one_qubit import decompose_one_qubit


def print_test_result(func_name, success, details=""):
    """打印测试结果"""
    status = "✓ SUCCESS" if success else "✗ FAILED"
    print(f"{func_name}: {status} {details}")


def test_decompose_one_qubit():
    """测试单量子比特门分解函数"""
    print("\n=== 测试单量子比特门分解函数 (decompose_one_qubit) ===")
    
    # 测试H门分解
    h_gate = HGate()
    try:
        # 分解为U基
        decomposed_h = decompose_one_qubit(h_gate, basis='U')
        print_test_result("H门分解为U基", True, f"深度: {decomposed_h.depth}, 门数: {len(decomposed_h.data)}")
        
        # 分解为ZXZ基
        decomposed_h_zxz = decompose_one_qubit(h_gate, basis='ZXZ')
        print_test_result("H门分解为ZXZ基", True, f"深度: {decomposed_h_zxz.depth}, 门数: {len(decomposed_h_zxz.data)}")
        
    except Exception as e:
        print_test_result("H门分解", False, str(e))
    
    # 测试X门分解
    x_gate = XGate()
    try:
        # 分解为U基
        decomposed_x = decompose_one_qubit(x_gate, basis='U')
        print_test_result("X门分解为U基", True, f"深度: {decomposed_x.depth}, 门数: {len(decomposed_x.data)}")
        
        # 分解为ZYZ基
        decomposed_x_zyz = decompose_one_qubit(x_gate, basis='ZYZ')
        print_test_result("X门分解为ZYZ基", True, f"深度: {decomposed_x_zyz.depth}, 门数: {len(decomposed_x_zyz.data)}")
        
    except Exception as e:
        print_test_result("X门分解", False, str(e))


if __name__ == "__main__":
    test_decompose_one_qubit()