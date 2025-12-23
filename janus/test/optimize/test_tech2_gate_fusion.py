"""
技术2: 基于量子门合并规则的电路优化 - 增强测试
目标：展示大规模单比特门合并和块合并的优化效果
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import unittest
import numpy as np
from circuit import Circuit as QuantumCircuit
from circuit import circuit_to_dag, dag_to_circuit
from circuit.library import UGate


class TestTech2Enhanced(unittest.TestCase):
    """技术2增强测试：门融合优化"""

    def test_1_enhanced_massive_single_qubit_fusion(self):
        """增强测试1: 大量连续单量子比特门合并"""
        from optimize import SingleQubitGateOptimizer, SingleQubitRunCollector

        n_qubits = 6
        qc = QuantumCircuit(n_qubits)

        # 每个量子比特上添加大量连续的单比特门
        for qubit in range(n_qubits):
            # 模拟参数化电路：每个量子比特10个连续的旋转门
            for i in range(10):
                theta = np.pi * (i + 1) / 10
                phi = np.pi * i / 8
                lam = np.pi * (i + 2) / 12
                qc.u(theta, phi, lam, qubit)

            # 添加更多可合并的门
            qc.rz(0.3, qubit)
            qc.rx(0.4, qubit)
            qc.ry(0.5, qubit)
            qc.rz(0.6, qubit)  # 这4个旋转门可以合并为1个U门

        # 添加一些CX门作为分隔（真实场景）
        for i in range(n_qubits - 1):
            qc.cx(i, i + 1)
            # CX后再加单比特门
            qc.u(0.1, 0.2, 0.3, i)
            qc.u(0.4, 0.5, 0.6, i + 1)
            qc.rz(0.7, i)
            qc.rx(0.8, i + 1)

        original_size = len(qc.data)
        original_depth = qc.depth()
        original_ops = qc.count_ops()

        # 应用优化
        dag = circuit_to_dag(qc)
        collector = SingleQubitRunCollector()
        dag = collector.run(dag)
        optimizer = SingleQubitGateOptimizer()
        dag_optimized = optimizer.run(dag)
        qc_optimized = dag_to_circuit(dag_optimized)

        optimized_size = len(qc_optimized.data)
        optimized_depth = qc_optimized.depth()

        print(f"\n测试1 - 大规模单比特门合并 ({n_qubits}量子比特):")
        print(f"  原始电路: {original_size}门, 深度{original_depth}")
        print(f"    U门: {original_ops.get('u', 0)}, Rz: {original_ops.get('rz', 0)}, "
              f"Rx: {original_ops.get('rx', 0)}, Ry: {original_ops.get('ry', 0)}")
        print(f"  优化电路: {optimized_size}门, 深度{optimized_depth}")
        print(f"  减少: {original_size - optimized_size}门 ({(original_size-optimized_size)/original_size*100:.1f}%)")

        self.assertLess(optimized_size, original_size * 0.6, "应至少减少40%的门")

    def test_2_enhanced_two_qubit_block_consolidation(self):
        """增强测试2: 复杂两量子比特块合并"""
        from optimize import TwoQubitBlockCollector, BlockConsolidator

        n_qubits = 4
        qc = QuantumCircuit(n_qubits)

        # 创建复杂的两量子比特交互模式
        for layer in range(3):
            # 每层包含多个两量子比特块
            for i in range(n_qubits - 1):
                # 复杂的两量子比特操作序列
                qc.cx(i, i + 1)
                qc.rz(0.3 * layer, i)
                qc.rx(0.4 * layer, i + 1)
                qc.cx(i, i + 1)
                qc.ry(0.5 * layer, i)
                qc.u(0.1, 0.2, 0.3, i + 1)
                qc.cx(i, i + 1)  # 3个CX + 单比特门可以合并

            # 反向连接
            for i in range(n_qubits - 1, 0, -1):
                qc.cx(i, i - 1)
                qc.u(0.6, 0.7, 0.8, i)
                qc.u(0.9, 1.0, 1.1, i - 1)
                qc.cx(i, i - 1)

        original_size = len(qc.data)
        original_depth = qc.depth()
        cx_count_original = qc.count_ops().get('cx', 0)

        # 应用优化
        dag = circuit_to_dag(qc)
        collector = TwoQubitBlockCollector()
        dag = collector.run(dag)
        consolidator = BlockConsolidator()
        dag_optimized = consolidator.run(dag)
        qc_optimized = dag_to_circuit(dag_optimized)

        optimized_size = len(qc_optimized.data)
        optimized_depth = qc_optimized.depth()
        cx_count_optimized = qc_optimized.count_ops().get('cx', 0)

        print(f"\n测试2 - 两量子比特块合并 ({n_qubits}量子比特, 3层):")
        print(f"  原始电路: {original_size}门, 深度{original_depth}, CX门{cx_count_original}个")
        print(f"  优化电路: {optimized_size}门, 深度{optimized_depth}, CX门{cx_count_optimized}个")
        print(f"  总门数减少: {original_size - optimized_size} ({(original_size-optimized_size)/original_size*100:.1f}%)")
        print(f"  CX门优化: {cx_count_original - cx_count_optimized} ({(cx_count_original-cx_count_optimized)/cx_count_original*100:.1f}%)")

        # 两量子比特块合并可能不会减少门数(如果已经是最优的)
        self.assertLessEqual(optimized_size, original_size)
        self.assertLessEqual(cx_count_optimized, cx_count_original)

    def test_3_enhanced_variational_circuit_optimization(self):
        """增强测试3: 变分量子电路优化（VQE/QAOA风格）"""
        from optimize import (
            SingleQubitRunCollector, SingleQubitGateOptimizer,
            TwoQubitBlockCollector, BlockConsolidator
        )

        n_qubits = 5
        n_layers = 4
        qc = QuantumCircuit(n_qubits)

        # 初始化层
        for i in range(n_qubits):
            qc.ry(np.pi/4, i)

        # 多层变分结构
        for layer in range(n_layers):
            # 纠缠层：全连接CX
            for i in range(n_qubits):
                for j in range(i + 1, n_qubits):
                    qc.cx(i, j)
                    qc.rz(0.1 * layer, i)
                    qc.rx(0.2 * layer, j)

            # 单比特旋转层（大量可合并门）
            for i in range(n_qubits):
                qc.ry(0.3 * layer, i)
                qc.rz(0.4 * layer, i)
                qc.rx(0.5 * layer, i)
                # 再来一组
                qc.u(0.1, 0.2, 0.3, i)
                qc.rz(0.6, i)
                qc.ry(0.7, i)

        original_size = len(qc.data)
        original_depth = qc.depth()

        # 完整优化流水线
        dag = circuit_to_dag(qc)

        # 单比特优化
        dag = SingleQubitRunCollector().run(dag)
        dag = SingleQubitGateOptimizer().run(dag)

        # 两比特优化
        dag = TwoQubitBlockCollector().run(dag)
        dag = BlockConsolidator().run(dag)

        qc_optimized = dag_to_circuit(dag)

        optimized_size = len(qc_optimized.data)
        optimized_depth = qc_optimized.depth()

        print(f"\n测试3 - 变分量子电路 ({n_qubits}比特, {n_layers}层):")
        print(f"  原始电路: {original_size}门, 深度{original_depth}")
        print(f"  优化电路: {optimized_size}门, 深度{optimized_depth}")
        reduction_pct = (original_size-optimized_size)/original_size*100 if original_size > 0 else 0
        depth_reduction_pct = (original_depth-optimized_depth)/original_depth*100 if original_depth > 0 else 0
        print(f"  优化效果: 减少{original_size - optimized_size}门 ({reduction_pct:.1f}%)")
        print(f"            深度减少{original_depth - optimized_depth} ({depth_reduction_pct:.1f}%)")

        self.assertLessEqual(optimized_size, original_size, "优化后不应增加门数")
        self.assertLessEqual(optimized_depth, original_depth, "优化后不应增加深度")


if __name__ == '__main__':
    print("="*70)
    print("技术2增强测试：门融合优化 - 显著优化效果展示")
    print("="*70)
    unittest.main(verbosity=2)
