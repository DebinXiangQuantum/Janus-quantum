# 测试双量子比特门分解函数
import sys
import os

# 添加项目根目录到sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import numpy as np

# 从Janus库导入必要组件
from circuit import Circuit
from circuit.library.standard_gates import CXGate, CZGate

# 导入分解函数
from janus.decompose.decompose_two_qubit_gate import decompose_two_qubit_gate


# 实现随机酉矩阵生成函数，替代Qiskit的random_unitary
def random_unitary(size):
    """生成随机酉矩阵"""
    # 使用QR分解方法生成随机酉矩阵
    from scipy.stats import unitary_group
    return unitary_group.rvs(size)


def print_test_result(func_name, success, details=""):
    """打印测试结果"""
    status = "✓ SUCCESS" if success else "✗ FAILED"
    print(f"{func_name}: {status} {details}")


def test_decompose_two_qubit_gate():
    """测试双量子比特门转换函数"""
    print("\n=== 测试双量子比特门转换函数 (decompose_two_qubit_gate) ===")
    
    # 测试CX门转换为不同基
    cx_gate = CXGate()
    
    # 只测试Janus库支持的基
    for basis in ['cx', 'cz']:
        try:
            decomposed_cx = decompose_two_qubit_gate(cx_gate, basis_gate=basis)
            # 获取电路深度和门数
            # 使用正确的Circuit API：depth是属性，_instructions包含所有指令
            depth = getattr(decomposed_cx, 'depth', 'N/A')
            gate_count = len(getattr(decomposed_cx, '_instructions', []))
            print_test_result(f"CX门转换为{basis}基", True, f"深度: {depth}, 门数: {gate_count}")
        except Exception as e:
            print_test_result(f"CX门转换为{basis}基", False, str(e))
    
    # 测试随机双量子比特门分解
    try:
        random_2q = random_unitary(4)
        decomposed_random = decompose_two_qubit_gate(random_2q, basis_gate='cx')
        # 获取电路深度和门数
        depth = getattr(decomposed_random, 'depth', 'N/A')
        gate_count = len(getattr(decomposed_random, '_instructions', []))
        print_test_result("随机双量子比特门分解", True, f"深度: {depth}, 门数: {gate_count}")
    except Exception as e:
        print_test_result("随机双量子比特门分解", False, str(e))


if __name__ == "__main__":
    test_decompose_two_qubit_gate()