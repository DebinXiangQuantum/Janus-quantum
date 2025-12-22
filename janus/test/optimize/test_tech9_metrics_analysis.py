"""
技术9: 电路资源指标分析 - 增强测试
目标：在复杂电路上展示资源分析的全面性和准确性
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import unittest
import numpy as np
from circuit import Circuit as QuantumCircuit
from circuit import circuit_to_dag, dag_to_circuit


class TestTech9Enhanced(unittest.TestCase):
    """技术9增强测试：电路资源分析"""

    def test_1_enhanced_large_circuit_analysis(self):
        """增强测试1: 大规模电路资源分析"""
        from optimize import CircuitResourceAnalyzer

        # 创建大规模变分量子电路
        n_qubits = 8
        n_layers = 5
        qc = QuantumCircuit(n_qubits)

        # 初始化
        for i in range(n_qubits):
            qc.h(i)

        # 多层纠缠结构
        for layer in range(n_layers):
            # 单比特旋转
            for i in range(n_qubits):
                qc.rx(0.1 * layer, i)
                qc.rz(0.2 * layer, i)
                qc.ry(0.3 * layer, i)

            # 纠缠层
            for i in range(n_qubits - 1):
                qc.cx(i, i + 1)
            qc.cx(n_qubits - 1, 0)  # 环形连接

        # 分析资源
        dag = circuit_to_dag(qc)
        analyzer = CircuitResourceAnalyzer()
        analyzer.run(dag)

        depth = analyzer.property_set.get('depth')
        width = analyzer.property_set.get('width')
        size = analyzer.property_set.get('size')
        count_ops = analyzer.property_set.get('count_ops')
        num_qubits = analyzer.property_set.get('num_qubits')

        print(f"\n测试1 - 大规模变分电路分析 ({n_qubits}比特, {n_layers}层):")
        print(f"  电路深度: {depth}")
        print(f"  电路宽度: {width}")
        print(f"  总门数: {size}")
        print(f"  量子比特数: {num_qubits}")
        print(f"  门类型分布: {count_ops}")
        print(f"  平均每层门数: {size/n_layers:.1f}")
        print(f"  平均每比特门数: {size/n_qubits:.1f}")
        print(f"  并行度: {size/depth if depth > 0 else 0:.2f}")

        self.assertIsNotNone(depth)
        self.assertIsNotNone(width)
        self.assertIsNotNone(size)
        self.assertEqual(width, n_qubits)

    def test_2_enhanced_optimization_impact_analysis(self):
        """增强测试2: 优化前后资源对比分析"""
        from optimize import (
            CircuitResourceAnalyzer,
            SingleQubitGateOptimizer,
            CommutativeGateCanceller
        )

        # 创建包含大量冗余的电路
        n_qubits = 6
        qc = QuantumCircuit(n_qubits)

        # 添加大量可优化的模式
        for _ in range(4):
            for i in range(n_qubits):
                # 多个可合并的单比特门
                qc.rz(0.1, i)
                qc.rx(0.2, i)
                qc.ry(0.3, i)
                qc.rz(0.4, i)

                # 可消除的门对
                qc.h(i)
                qc.h(i)
                qc.t(i)
                qc.tdg(i)

            # CX层
            for i in range(n_qubits - 1):
                qc.cx(i, i + 1)

        # 分析原始电路
        dag = circuit_to_dag(qc)
        analyzer_original = CircuitResourceAnalyzer()
        analyzer_original.run(dag)

        original_metrics = {
            'depth': analyzer_original.property_set.get('depth'),
            'size': analyzer_original.property_set.get('size'),
            'ops': analyzer_original.property_set.get('count_ops')
        }

        # 应用优化
        dag_opt = SingleQubitGateOptimizer().run(dag)
        dag_opt = CommutativeGateCanceller().run(dag_opt)

        # 分析优化后电路
        analyzer_optimized = CircuitResourceAnalyzer()
        analyzer_optimized.run(dag_opt)

        optimized_metrics = {
            'depth': analyzer_optimized.property_set.get('depth'),
            'size': analyzer_optimized.property_set.get('size'),
            'ops': analyzer_optimized.property_set.get('count_ops')
        }

        # 计算改进
        size_reduction = original_metrics['size'] - optimized_metrics['size']
        depth_reduction = original_metrics['depth'] - optimized_metrics['depth']

        print(f"\n测试2 - 优化影响分析 ({n_qubits}量子比特):")
        print(f"  原始电路:")
        print(f"    深度: {original_metrics['depth']}, 大小: {original_metrics['size']}")
        print(f"    操作: {original_metrics['ops']}")
        print(f"  优化电路:")
        print(f"    深度: {optimized_metrics['depth']}, 大小: {optimized_metrics['size']}")
        print(f"    操作: {optimized_metrics['ops']}")
        print(f"  改进:")
        print(f"    大小减少: {size_reduction} ({size_reduction/original_metrics['size']*100:.1f}%)")
        print(f"    深度减少: {depth_reduction} ({depth_reduction/original_metrics['depth']*100:.1f}%)")

        self.assertLess(optimized_metrics['size'], original_metrics['size'])

    def test_3_enhanced_algorithm_complexity_comparison(self):
        """增强测试3: 不同量子算法的资源需求对比"""
        from optimize import CircuitResourceAnalyzer

        algorithms = {}

        # 算法1: QFT (Quantum Fourier Transform)
        n = 5
        qc_qft = QuantumCircuit(n)
        for i in range(n):
            qc_qft.h(i)
            for j in range(i + 1, n):
                qc_qft.cp(np.pi / (2 ** (j - i)), j, i)

        dag = circuit_to_dag(qc_qft)
        analyzer = CircuitResourceAnalyzer()
        analyzer.run(dag)
        algorithms['QFT'] = {
            'depth': analyzer.property_set.get('depth'),
            'size': analyzer.property_set.get('size'),
            'cx': analyzer.property_set.get('count_ops', {}).get('cx', 0)
        }

        # 算法2: Grover搜索（单次迭代）
        qc_grover = QuantumCircuit(n)
        # 初始化
        for i in range(n):
            qc_grover.h(i)
        # Oracle（简化）
        qc_grover.x(n - 1)
        for i in range(n - 1):
            qc_grover.cx(i, n - 1)
        qc_grover.x(n - 1)
        # Diffusion
        for i in range(n):
            qc_grover.h(i)
            qc_grover.x(i)
        qc_grover.h(n - 1)
        for i in range(n - 1):
            qc_grover.cx(i, n - 1)
        qc_grover.h(n - 1)
        for i in range(n):
            qc_grover.x(i)
            qc_grover.h(i)

        dag = circuit_to_dag(qc_grover)
        analyzer = CircuitResourceAnalyzer()
        analyzer.run(dag)
        algorithms['Grover'] = {
            'depth': analyzer.property_set.get('depth'),
            'size': analyzer.property_set.get('size'),
            'cx': analyzer.property_set.get('count_ops', {}).get('cx', 0)
        }

        # 算法3: QAOA (单层)
        qc_qaoa = QuantumCircuit(n)
        for i in range(n):
            qc_qaoa.h(i)
        # 问题哈密顿量
        for i in range(n - 1):
            qc_qaoa.cx(i, i + 1)
            qc_qaoa.rz(0.5, i + 1)
            qc_qaoa.cx(i, i + 1)
        # 混合哈密顿量
        for i in range(n):
            qc_qaoa.rx(0.3, i)

        dag = circuit_to_dag(qc_qaoa)
        analyzer = CircuitResourceAnalyzer()
        analyzer.run(dag)
        algorithms['QAOA'] = {
            'depth': analyzer.property_set.get('depth'),
            'size': analyzer.property_set.get('size'),
            'cx': analyzer.property_set.get('count_ops', {}).get('cx', 0)
        }

        print(f"\n测试3 - 量子算法复杂度对比 ({n}量子比特):")
        print(f"  {'算法':<10} {'深度':>6} {'总门数':>8} {'CX门':>6}")
        print(f"  {'-'*10} {'-'*6} {'-'*8} {'-'*6}")
        for name, metrics in algorithms.items():
            print(f"  {name:<10} {metrics['depth']:>6} {metrics['size']:>8} {metrics['cx']:>6}")

        self.assertTrue(all(m['depth'] is not None for m in algorithms.values()))


if __name__ == '__main__':
    print("="*70)
    print("技术9增强测试：电路资源分析 - 全面资源评估")
    print("="*70)
    unittest.main(verbosity=2)
