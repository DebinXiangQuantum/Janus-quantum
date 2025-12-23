# 测试多控制Toffoli门分解函数
import sys
import os

# 添加项目根目录到sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import numpy as np
from circuit.circuit import Circuit
from circuit.library.standard_gates import XGate

# 导入分解函数
from janus.decompose.decompose_multi_control_toffoli import decompose_multi_control_toffoli


def print_test_result(func_name, success, details=""):
    """打印测试结果"""
    status = "✓ SUCCESS" if success else "✗ FAILED"
    print(f"{func_name}: {status} {details}")


def test_decompose_multi_control_toffoli():
    """测试多控制Toffoli门分解函数"""
    print("\n=== 测试多控制Toffoli门分解函数 (decompose_multi_control_toffoli) ===")
    
    # 测试不同数量的控制量子比特
    for num_ctrl in [1, 2, 3, 4]:
        try:
            # 使用hp24方法（仅支持该方法）
            decomposed_mct = decompose_multi_control_toffoli(num_ctrl_qubits=num_ctrl)
            print_test_result(f"{num_ctrl}控制Toffoli门分解", True, f"深度: {decomposed_mct.depth}, 门数: {len(decomposed_mct.instructions)}")
            
        except Exception as e:
            print_test_result(f"{num_ctrl}控制Toffoli门分解", False, str(e))


if __name__ == "__main__":
    test_decompose_multi_control_toffoli()