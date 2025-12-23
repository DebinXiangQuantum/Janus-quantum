"""
技术5: 局部子电路的KAK分解优化 - 增强测试

测试目标:
- 验证双比特门KAK(Cartan)分解
- 验证Weyl分解准确性
- 验证ConsolidateBlocks中的KAK优化（功能5.4）
- 展示显著的优化效果
"""

import sys
import os

# 添加janus目录到Python路径（使用optimize模块）
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


import unittest
import numpy as np
from circuit import Circuit as QuantumCircuit
from qiskit.quantum_info import random_unitary, Operator
from circuit import circuit_to_dag, dag_to_circuit


class TestTech5KAKDecomposeOptimization(unittest.TestCase):
    """技术5: 局部子电路的KAK分解优化测试类"""

    def test_5_1_kak_decomposition_basic(self):
        """测试5.1: KAK分解基础功能"""
        from optimize import KAKDecomposition

        unitary = random_unitary(4, seed=42).data
        decomp = KAKDecomposition(unitary)

        self.assertIsNotNone(decomp.a)
        print(f"\n测试5.1通过: KAK分解成功")
        print(f"   Weyl坐标: a={decomp.a:.4f}, b={decomp.b:.4f}, c={decomp.c:.4f}")

    def test_5_2_kak_basis_decomposition(self):
        """测试5.2: KAK基础门集分解"""
        from optimize import KAKBasisDecomposer
        from circuit.library import CXGate

        decomposer = KAKBasisDecomposer(CXGate())
        unitary = random_unitary(4, seed=123).data
        circuit = decomposer(unitary)

        original_op = Operator(unitary)
        decomposed_op = Operator(circuit)
        fidelity = np.abs(np.trace(original_op.adjoint().dot(decomposed_op)) / 4)

        print(f"\n测试5.2通过: KAK基础门集分解成功")
        print(f"   使用CX门数: {circuit.count_ops().get('cx', 0)}")
        print(f"   保真度: {fidelity:.10f}")
        self.assertGreater(fidelity, 0.9999)

    def test_5_3_kak_decomposition_precision(self):
        """测试5.3: KAK分解精度验证"""
        from optimize import KAKBasisDecomposer
        from circuit.library import CXGate

        precisions = []
        for seed in range(10):
            unitary = random_unitary(4, seed=seed).data
            decomposer = KAKBasisDecomposer(CXGate())
            circuit = decomposer(unitary)

            original_op = Operator(unitary)
            decomposed_op = Operator(circuit)
            error = np.linalg.norm(original_op.data - decomposed_op.data)
            precisions.append(error)

        avg_error = np.mean(precisions)
        max_error = np.max(precisions)

        print(f"\n测试5.3通过: KAK分解精度验证")
        print(f"   平均误差: {avg_error:.2e}")
        print(f"   最大误差: {max_error:.2e}")
        self.assertLess(max_error, 1e-10)

    def test_5_4_consolidate_blocks_kak_optimization(self):
        """测试5.4: ConsolidateBlocks中的KAK分解优化（展示优化效果）"""
        from optimize import (
            TwoQubitBlockCollector,
            BlockConsolidator
        )
        from circuit.library import CXGate

        # 创建包含多个两量子比特门的复杂电路
        # 这些门可以通过KAK分解优化合并
        qc = QuantumCircuit(4)

        # 在q0-q1上创建可优化的两比特门序列
        qc.cx(0, 1)
        qc.u(0.1, 0.2, 0.3, 0)
        qc.u(0.4, 0.5, 0.6, 1)
        qc.cx(0, 1)
        qc.u(0.15, 0.25, 0.35, 0)
        qc.u(0.45, 0.55, 0.65, 1)
        qc.cx(0, 1)

        # 在q2-q3上创建另一组
        qc.cx(2, 3)
        qc.rz(0.5, 2)
        qc.rx(0.3, 3)
        qc.cx(2, 3)
        qc.rz(0.7, 2)
        qc.rx(0.4, 3)
        qc.cx(2, 3)

        original_cx_count = qc.count_ops().get('cx', 0)
        original_size = len(qc.data)
        original_depth = qc.depth()

        # 使用ConsolidateBlocks优化（内部使用KAK分解）
        dag = circuit_to_dag(qc)

        # 步骤1: 收集两量子比特块
        collector = TwoQubitBlockCollector()
        collector.run(dag)

        # 步骤2: 使用KAK分解合并块（传入Gate对象）
        consolidator = BlockConsolidator(kak_basis_gate=CXGate())
        dag_opt = consolidator.run(dag)

        qc_opt = dag_to_circuit(dag_opt)

        optimized_cx_count = qc_opt.count_ops().get('cx', 0)
        optimized_size = len(qc_opt.data)
        optimized_depth = qc_opt.depth()

        cx_reduction = original_cx_count - optimized_cx_count
        size_reduction = original_size - optimized_size
        depth_reduction = original_depth - optimized_depth

        print(f"\n测试5.4通过: ConsolidateBlocks KAK优化（功能5.4）")
        print(f"   原始电路:")
        print(f"     - 总门数: {original_size}")
        print(f"     - CX门数: {original_cx_count}")
        print(f"     - 深度: {original_depth}")
        print(f"   KAK优化后:")
        print(f"     - 总门数: {optimized_size} (减少{size_reduction}, {size_reduction/original_size*100:.1f}%)")
        print(f"     - CX门数: {optimized_cx_count} (减少{cx_reduction}, {cx_reduction/original_cx_count*100:.1f}%)")
        print(f"     - 深度: {optimized_depth} (减少{depth_reduction}, {depth_reduction/original_depth*100:.1f}%)")
        print(f"   使用Janus模块: BlockConsolidator (内部使用KAK分解)")

        # 验证优化效果
        self.assertLess(optimized_cx_count, original_cx_count, "KAK优化应减少CX门数")
        self.assertLess(optimized_size, original_size, "应减少总门数")

    def test_5_5_large_scale_two_qubit_kak_optimization(self):
        """测试5.5: 大规模两量子比特电路KAK优化"""
        from optimize import (
            TwoQubitBlockCollector,
            BlockConsolidator
        )
        from circuit.library import CXGate

        # 创建包含大量两量子比特门的电路（模拟变分算法）
        qc = QuantumCircuit(6)

        # 4层变分电路结构
        for layer in range(4):
            # 纠缠层：相邻比特CX
            for i in range(5):
                qc.cx(i, i+1)

            # 单比特旋转层
            for i in range(6):
                qc.rx(0.1 * (layer + 1), i)
                qc.rz(0.2 * (layer + 1), i)

            # 反向纠缠层
            for i in range(4, -1, -1):
                qc.cx(i, i+1)

        original_cx_count = qc.count_ops().get('cx', 0)
        original_size = len(qc.data)
        original_depth = qc.depth()

        # KAK优化
        dag = circuit_to_dag(qc)
        collector = TwoQubitBlockCollector()
        collector.run(dag)
        consolidator = BlockConsolidator(kak_basis_gate=CXGate())
        dag_opt = consolidator.run(dag)
        qc_opt = dag_to_circuit(dag_opt)

        optimized_cx_count = qc_opt.count_ops().get('cx', 0)
        optimized_size = len(qc_opt.data)
        optimized_depth = qc_opt.depth()

        print(f"\n测试5.5通过: 大规模变分电路KAK优化")
        print(f"   电路规模: 6量子比特, 4层变分结构")
        print(f"   原始: CX={original_cx_count}, 总门数={original_size}, 深度={original_depth}")
        print(f"   优化: CX={optimized_cx_count}, 总门数={optimized_size}, 深度={optimized_depth}")
        print(f"   优化率: CX减少{(original_cx_count-optimized_cx_count)/original_cx_count*100:.1f}%, "
              f"总门数减少{(original_size-optimized_size)/original_size*100:.1f}%")

        self.assertLessEqual(optimized_cx_count, original_cx_count)
        self.assertLessEqual(optimized_size, original_size)


if __name__ == '__main__':
    print("="*70)
    print("技术5增强测试: KAK分解优化 - ConsolidateBlocks优化效果展示")
    print("="*70)
    unittest.main(verbosity=2)
