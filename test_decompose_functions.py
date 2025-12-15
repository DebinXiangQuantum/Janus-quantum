# 量子门分解函数测试脚本
# 测试DECOMPOSE_FUNCTION_LOCATIONS.md中记录的所有分解函数

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, random_unitary
from qiskit.circuit.library import (
    XGate, YGate, ZGate, HGate, CXGate, CZGate, SwapGate, RXXGate, RXGate, RYGate, RZGate
)

# 导入所有分解函数
from qiskit.synthesis.one_qubit import decompose_one_qubit
from qiskit.synthesis.two_qubit import decompose_two_qubit_gate
from qiskit.synthesis.multi_controlled import decompose_multi_control_toffoli
from qiskit.circuit.library.standard_gates import decompose_controlled_gate, decompose_kak
from circuit_to_instruction_set import convert_circuit_to_instruction_set

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
        # 分解为U3基
        decomposed_h = decompose_one_qubit(h_gate, basis='U3')
        print_test_result("H门分解为U3基", True, f"深度: {decomposed_h.depth()}, 门数: {len(decomposed_h.data)}")
        
        # 分解为ZXZ基
        decomposed_h_zxz = decompose_one_qubit(h_gate, basis='ZXZ')
        print_test_result("H门分解为ZXZ基", True, f"深度: {decomposed_h_zxz.depth()}, 门数: {len(decomposed_h_zxz.data)}")
        
    except Exception as e:
        print_test_result("H门分解", False, str(e))
    
    # 测试随机单量子比特门分解
    random_1q = random_unitary(2)
    try:
        decomposed_random = decompose_one_qubit(random_1q, basis='U')
        print_test_result("随机单量子比特门分解", True, f"深度: {decomposed_random.depth()}, 门数: {len(decomposed_random.data)}")
    except Exception as e:
        print_test_result("随机单量子比特门分解", False, str(e))

def test_decompose_two_qubit_gate():
    """测试双量子比特门转换函数"""
    print("\n=== 测试双量子比特门转换函数 (decompose_two_qubit_gate) ===")
    
    # 测试CX门转换为不同基
    cx_gate = CXGate()
    
    for basis in ['cx', 'cz', 'rxx', 'rzz']:
        try:
            decomposed_cx = decompose_two_qubit_gate(cx_gate, basis_gate=basis)
            print_test_result(f"CX门转换为{basis}基", True, f"深度: {decomposed_cx.depth()}, 门数: {len(decomposed_cx.data)}")
        except Exception as e:
            print_test_result(f"CX门转换为{basis}基", False, str(e))
    
    # 测试随机双量子比特门分解
    random_2q = random_unitary(4)
    try:
        decomposed_random = decompose_two_qubit_gate(random_2q, basis_gate='cx')
        print_test_result("随机双量子比特门分解", True, f"深度: {decomposed_random.depth()}, 门数: {len(decomposed_random.data)}")
    except Exception as e:
        print_test_result("随机双量子比特门分解", False, str(e))

def test_decompose_multi_control_toffoli():
    """测试多控制Toffoli门分解函数"""
    print("\n=== 测试多控制Toffoli门分解函数 (decompose_multi_control_toffoli) ===")
    
    # 测试不同数量的控制量子比特
    for num_ctrl in [2, 3, 4]:
        try:
            # 使用默认方法
            decomposed_mct = decompose_multi_control_toffoli(num_ctrl_qubits=num_ctrl)
            print_test_result(f"{num_ctrl}控制Toffoli门分解(默认方法)", True, f"深度: {decomposed_mct.depth()}, 门数: {len(decomposed_mct.data)}")
            
            # 使用hp24方法（高效无辅助）
            decomposed_mct_hp24 = decompose_multi_control_toffoli(num_ctrl_qubits=num_ctrl, method='hp24')
            print_test_result(f"{num_ctrl}控制Toffoli门分解(hp24方法)", True, f"深度: {decomposed_mct_hp24.depth()}, 门数: {len(decomposed_mct_hp24.data)}")
            
        except Exception as e:
            print_test_result(f"{num_ctrl}控制Toffoli门分解", False, str(e))
    
    # 测试使用辅助量子比特的分解
    try:
        decomposed_mct_ancilla = decompose_multi_control_toffoli(num_ctrl_qubits=5, method='kg24', num_ancilla_qubits=2)
        print_test_result("5控制Toffoli门分解(kg24+2辅助)", True, f"深度: {decomposed_mct_ancilla.depth()}, 门数: {len(decomposed_mct_ancilla.data)}")
    except Exception as e:
        print_test_result("5控制Toffoli门分解(kg24+2辅助)", False, str(e))

def test_decompose_controlled_gate():
    """测试受控门分解函数"""
    print("\n=== 测试受控门分解函数 (decompose_controlled_gate) ===")
    
    # 测试不同的基础门
    non_x_gates = [
        (YGate(), "YGate"),
        (ZGate(), "ZGate"),
        (RXGate(0.5), "RXGate"),
        (RYGate(0.5), "RYGate"),
        (RZGate(0.5), "RZGate")
    ]
    
    # 单独测试XGate（使用hp24方法，这个方法在两个函数中都有定义）
    try:
        decomposed_x = decompose_controlled_gate(XGate(), num_ctrl_qubits=2, method='hp24')
        print_test_result("2控制XGate分解(hp24)", True, f"深度: {decomposed_x.depth()}, 门数: {len(decomposed_x.data)}")
    except Exception as e:
        print_test_result("2控制XGate分解", False, str(e))
    
    # 测试其他门
    for gate, gate_name in non_x_gates:
        try:
            # 2控制门分解 - 使用default方法确保兼容性
            decomposed_gate = decompose_controlled_gate(gate, num_ctrl_qubits=2, method='default')
            print_test_result(f"2控制{gate_name}分解(default)", True, f"深度: {decomposed_gate.depth()}, 门数: {len(decomposed_gate.data)}")
        except Exception as e:
            print_test_result(f"2控制{gate_name}分解", False, str(e))
    
    # 测试3控制X门的高效分解
    try:
        decomposed_3cx = decompose_controlled_gate(XGate(), num_ctrl_qubits=3, method='hp24')
        print_test_result("3控制X门分解(hp24)", True, f"深度: {decomposed_3cx.depth()}, 门数: {len(decomposed_3cx.data)}")
    except Exception as e:
        print_test_result("3控制X门分解(hp24)", False, str(e))

def test_decompose_kak():
    """测试任意数量量子比特KAK分解函数"""
    print("\n=== 测试任意数量量子比特KAK分解函数 (decompose_kak) ===")
    
    # 测试1量子比特门分解
    h_gate = HGate()
    try:
        decomposed_h = decompose_kak(h_gate, euler_basis='U3', simplify=True)
        print_test_result("1量子比特H门KAK分解", True, f"深度: {decomposed_h.depth()}, 门数: {len(decomposed_h.data)}")
    except Exception as e:
        print_test_result("1量子比特H门KAK分解", False, str(e))
    
    # 测试2量子比特门分解
    two_qubit_gates = [
        (CXGate(), "CXGate"),
        (CZGate(), "CZGate"),
        (SwapGate(), "SwapGate"),
        (RXXGate(0.5), "RXXGate")
    ]
    
    for gate, gate_name in two_qubit_gates:
        try:
            decomposed_gate = decompose_kak(gate, euler_basis='U3', simplify=True)
            print_test_result(f"2量子比特{gate_name}KAK分解", True, f"深度: {decomposed_gate.depth()}, 门数: {len(decomposed_gate.data)}")
        except Exception as e:
            print_test_result(f"2量子比特{gate_name}KAK分解", False, str(e))
    
    # 测试3量子比特随机幺正矩阵分解
    try:
        random_3q = random_unitary(8)
        decomposed_3q = decompose_kak(random_3q, euler_basis='U3', simplify=True)
        print_test_result("3量子比特随机幺正矩阵KAK分解", True, f"深度: {decomposed_3q.depth()}, 门数: {len(decomposed_3q.data)}")
    except Exception as e:
        print_test_result("3量子比特随机幺正矩阵KAK分解", False, str(e))

def test_convert_circuit_to_instruction_set():
    """测试电路到指令集转换函数"""
    print("\n=== 测试电路到指令集转换函数 (convert_circuit_to_instruction_set) ===")
    
    # 创建测试电路
    def create_test_circuit(num_qubits=2):
        qc = QuantumCircuit(num_qubits)
        qc.h(0)
        for i in range(num_qubits - 1):
            qc.cx(i, i + 1)
        qc.z(range(num_qubits))
        qc.x(0)
        qc.y(num_qubits - 1)
        return qc
    
    # 测试2量子比特电路
    qc2 = create_test_circuit(2)
    
    # 测试不同的指令集
    instruction_sets = [
        (['u3', 'cx'], "u3 + cx"),
        (['u', 'cx'], "u + cx"),
        (['u', 'cx', 'rz'], "u + cx + rz")
    ]
    
    for inst_set, inst_set_name in instruction_sets:
        try:
            converted_qc = convert_circuit_to_instruction_set(qc2, inst_set)
            gate_counts = converted_qc.count_ops()
            print_test_result(f"电路转换为{inst_set_name}指令集", True, f"深度: {converted_qc.depth()}, 门数: {gate_counts}")
        except Exception as e:
            print_test_result(f"电路转换为{inst_set_name}指令集", False, str(e))
    
    # 测试带耦合映射的转换
    try:
        qc3 = create_test_circuit(3)
        coupling_map = [[0, 1], [1, 2]]  # 线性链
        converted_qc3 = convert_circuit_to_instruction_set(
            qc3, ['u3', 'cx'], coupling_map=coupling_map, optimization_level=2
        )
        gate_counts = converted_qc3.count_ops()
        print_test_result("3量子比特电路转换(带耦合映射)", True, f"深度: {converted_qc3.depth()}, 门数: {gate_counts}")
    except Exception as e:
        print_test_result("3量子比特电路转换(带耦合映射)", False, str(e))

def main():
    """主测试函数"""
    print("量子门分解函数测试开始")
    print("=" * 50)
    
    # 运行所有测试
    test_decompose_one_qubit()
    test_decompose_two_qubit_gate()
    test_decompose_multi_control_toffoli()
    test_decompose_controlled_gate()
    test_decompose_kak()
    test_convert_circuit_to_instruction_set()
    
    print("\n" + "=" * 50)
    print("量子门分解函数测试结束")

if __name__ == "__main__":
    main()
