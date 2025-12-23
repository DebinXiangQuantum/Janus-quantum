"""
技术3: 基于交换性的门消除优化 - 增强测试
目标：展示通过门交换和消除实现的显著优化效果
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import unittest
from circuit import Circuit as QuantumCircuit
from circuit import circuit_to_dag, dag_to_circuit


class TestTech3Enhanced(unittest.TestCase):
    """技术3增强测试：交换性优化"""

    def test_1_enhanced_massive_self_adjoint_cancellation(self):
        """增强测试1: 大量自伴门消除（H·H=I, X·X=I, Y·Y=I, Z·Z=I, CX·CX=I）"""
        from optimize import CommutativeGateCanceller

        n_qubits = 6
        qc = QuantumCircuit(n_qubits)

        # 创建大量可消除的自伴门对
        for repeat in range(5):
            for i in range(n_qubits):
                qc.h(i)
                qc.x(i)
                qc.y(i)
                qc.z(i)
                # 反向添加相同的门（自伴）
                qc.z(i)
                qc.y(i)
                qc.x(i)
                qc.h(i)  # 应该全部消除

            # CX门对
            for i in range(n_qubits - 1):
                qc.cx(i, i + 1)
                qc.cx(i, i + 1)  # CX·CX = I

        # 添加不可消除的门作为对比
        for i in range(n_qubits):
            qc.t(i)

        original_size = len(qc.data)
        original_ops = qc.count_ops()

        # 应用优化
        dag = circuit_to_dag(qc)
        optimizer = CommutativeGateCanceller()
        dag_optimized = optimizer.run(dag)
        qc_optimized = dag_to_circuit(dag_optimized)

        optimized_size = len(qc_optimized.data)
        optimized_ops = qc_optimized.count_ops()

        print(f"\n测试1 - 大规模自伴门消除 ({n_qubits}量子比特, 5轮):")
        print(f"  原始电路: {original_size}门")
        print(f"    H: {original_ops.get('h', 0)}, X: {original_ops.get('x', 0)}, "
              f"Y: {original_ops.get('y', 0)}, Z: {original_ops.get('z', 0)}, "
              f"CX: {original_ops.get('cx', 0)}")
        print(f"  优化电路: {optimized_size}门")
        print(f"  消除: {original_size - optimized_size}门 ({(original_size-optimized_size)/original_size*100:.1f}%)")

        self.assertLess(optimized_size, original_size, "优化应减少门数")

    def test_2_enhanced_inverse_gate_cancellation(self):
        """增强测试2: 大量互逆门对消除（T·Tdg=I, S·Sdg=I, Rx·Rx†=I）"""
        from optimize import InverseGateCanceller

        n_qubits = 6
        qc = QuantumCircuit(n_qubits)

        # 创建大量互逆门对
        for cycle in range(4):
            for i in range(n_qubits):
                # T和Tdg对
                qc.t(i)
                qc.tdg(i)  # 应消除

                # S和Sdg对
                qc.s(i)
                qc.sdg(i)  # 应消除

                # Rx旋转和反旋转
                qc.rx(0.5, i)
                qc.rx(-0.5, i)  # 应消除

                # Ry旋转和反旋转
                qc.ry(0.7, i)
                qc.ry(-0.7, i)  # 应消除

                # Rz旋转和反旋转
                qc.rz(0.3, i)
                qc.rz(-0.3, i)  # 应消除

        # 交错的模式
        for i in range(n_qubits - 1):
            qc.cx(i, i + 1)
            qc.t(i)
            qc.tdg(i)  # 可消除
            qc.s(i + 1)
            qc.sdg(i + 1)  # 可消除
            qc.cx(i, i + 1)

        original_size = len(qc.data)
        original_ops = qc.count_ops()

        # 应用优化
        dag = circuit_to_dag(qc)
        optimizer = InverseGateCanceller()
        dag_optimized = optimizer.run(dag)
        qc_optimized = dag_to_circuit(dag_optimized)

        optimized_size = len(qc_optimized.data)

        print(f"\n测试2 - 大规模互逆门消除 ({n_qubits}量子比特, 4轮):")
        print(f"  原始电路: {original_size}门")
        print(f"    T: {original_ops.get('t', 0)}, Tdg: {original_ops.get('tdg', 0)}")
        print(f"    S: {original_ops.get('s', 0)}, Sdg: {original_ops.get('sdg', 0)}")
        print(f"    Rx: {original_ops.get('rx', 0)}, Ry: {original_ops.get('ry', 0)}, Rz: {original_ops.get('rz', 0)}")
        print(f"  优化电路: {optimized_size}门")
        print(f"  消除: {original_size - optimized_size}门 ({(original_size-optimized_size)/original_size*100:.1f}%)")

        self.assertLess(optimized_size, original_size * 0.8, "应消除大量互逆门对")

    def test_3_enhanced_commutative_cancellation_with_swaps(self):
        """增强测试3: 基于交换性的复杂门消除"""
        from optimize import (
            GateCommutationAnalyzer,
            CommutativeGateCanceller,
            InverseGateCanceller,
            CommutativeInverseGateCanceller
        )

        n_qubits = 5
        qc = QuantumCircuit(n_qubits)

        # 创建复杂的可通过交换消除的模式
        for layer in range(3):
            # 模式1: Z门和其他门的交换
            for i in range(n_qubits):
                qc.h(i)
                qc.z(i)  # Z门
                qc.t(i)  # T和Z交换
                qc.z(i)  # 两个Z可以消除
                qc.s(i)  # S和Z交换
                qc.h(i)  # H-H可以消除

            # 模式2: 可交换的CX门和单比特门
            for i in range(n_qubits - 1):
                qc.cx(i, i + 1)
                qc.z(i + 1)  # Z在CX目标上
                qc.cx(i, i + 1)  # CX-Z-CX可以优化
                qc.x(i)
                qc.x(i)  # X-X消除

            # 模式3: 相位门的累积和消除
            for i in range(n_qubits):
                qc.t(i)
                qc.s(i)
                qc.tdg(i)  # T和Tdg在中间有S，可能通过交换消除
                qc.sdg(i)  # S和Sdg消除

        original_size = len(qc.data)
        original_depth = qc.depth()

        # 完整优化流水线
        dag = circuit_to_dag(qc)

        # 分析交换关系
        dag = GateCommutationAnalyzer().run(dag)

        # 应用所有消除优化
        dag = CommutativeGateCanceller().run(dag)
        dag = InverseGateCanceller().run(dag)
        dag = CommutativeInverseGateCanceller().run(dag)

        qc_optimized = dag_to_circuit(dag)

        optimized_size = len(qc_optimized.data)
        optimized_depth = qc_optimized.depth()

        print(f"\n测试3 - 基于交换性的复杂门消除 ({n_qubits}量子比特, 3层):")
        print(f"  原始电路: {original_size}门, 深度{original_depth}")
        print(f"  优化电路: {optimized_size}门, 深度{optimized_depth}")
        print(f"  门数减少: {original_size - optimized_size} ({(original_size-optimized_size)/original_size*100:.1f}%)")
        print(f"  深度减少: {original_depth - optimized_depth} ({(original_depth-optimized_depth)/original_depth*100:.1f}%)")

        self.assertLess(optimized_size, original_size * 0.6, "应减少至少40%的门")

    def test_4_enhanced_quantum_error_correction_pattern(self):
        """增强测试4: 量子纠错电路中的对称模式消除"""
        from optimize import (
            CommutativeGateCanceller,
            InverseGateCanceller
        )

        # 模拟3量子比特bit-flip纠错码的编码和解码
        qc = QuantumCircuit(5)

        # 编码电路
        qc.cx(0, 1)
        qc.cx(0, 2)

        # 模拟噪声（应用门）
        for i in range(3):
            qc.h(i)
            qc.t(i)
            qc.s(i)

        # 错误综合征测量准备
        qc.cx(0, 3)
        qc.cx(1, 3)
        qc.cx(1, 4)
        qc.cx(2, 4)

        # 再测量（反向操作，产生大量对称）
        qc.cx(2, 4)
        qc.cx(1, 4)
        qc.cx(1, 3)
        qc.cx(0, 3)

        # 解码（反向编码）
        qc.cx(0, 2)
        qc.cx(0, 1)

        # 对称的噪声门（应该大量消除）
        for i in range(3):
            qc.s(i)
            qc.t(i)
            qc.h(i)

        # 重复整个模式3次
        original_qc = qc.copy()
        for _ in range(2):
            qc.compose(original_qc, inplace=True)

        original_size = len(qc.data)
        cx_original = qc.count_ops().get('cx', 0)

        # 应用优化
        dag = circuit_to_dag(qc)
        dag = CommutativeGateCanceller().run(dag)
        dag = InverseGateCanceller().run(dag)
        qc_optimized = dag_to_circuit(dag)

        optimized_size = len(qc_optimized.data)
        cx_optimized = qc_optimized.count_ops().get('cx', 0)

        print(f"\n测试4 - 纠错电路对称模式消除:")
        print(f"  原始电路: {original_size}门 (CX: {cx_original})")
        print(f"  优化电路: {optimized_size}门 (CX: {cx_optimized})")
        print(f"  减少: {original_size - optimized_size}门 ({(original_size-optimized_size)/original_size*100:.1f}%)")

        self.assertLess(optimized_size, original_size * 0.9)


if __name__ == '__main__':
    print("="*70)
    print("技术3增强测试：交换性门消除优化 - 显著优化效果展示")
    print("="*70)
    unittest.main(verbosity=2)
