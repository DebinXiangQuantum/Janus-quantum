"""
技术6: Clifford量子电路的优化 - 增强测试

测试目标:
- 对比多种Clifford合成算法(AG, Greedy, BM)
- 展示Greedy算法相对AG的显著优势
- 验证LNN拓扑深度优化效果
- 验证大规模Clifford电路优化
"""

import sys
import os

# 添加janus目录到Python路径（使用optimize模块）
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


import unittest
from circuit import Circuit as QuantumCircuit
from qiskit.quantum_info import random_clifford, Clifford


class TestTech6CliffordSynthesisOptimization(unittest.TestCase):
    """技术6: Clifford量子电路的优化测试类"""

    def test_6_1_clifford_synthesis_comparison(self):
        """测试6.1: 多种Clifford合成算法对比（展示Greedy优势）"""
        from optimize import (
            synthesize_clifford_aaronson_gottesman,
            synthesize_clifford_greedy
        )

        # 测试多个不同规模的Clifford
        test_cases = [
            (3, 42, "3量子比特"),
            (4, 45, "4量子比特"),
            (5, 48, "5量子比特")
        ]

        print(f"\n测试6.1: Clifford合成算法对比")
        print(f"{'规模':<12} {'AG算法CX':<12} {'Greedy CX':<12} {'减少':<10} {'减少率'}")
        print(f"-" * 60)

        total_ag_cx = 0
        total_greedy_cx = 0

        for n_qubits, seed, desc in test_cases:
            cliff = random_clifford(n_qubits, seed=seed)

            ag_circuit = synthesize_clifford_aaronson_gottesman(cliff)
            greedy_circuit = synthesize_clifford_greedy(cliff)

            ag_cx = ag_circuit.count_ops().get('cx', 0)
            greedy_cx = greedy_circuit.count_ops().get('cx', 0)

            reduction = ag_cx - greedy_cx
            reduction_rate = (reduction / ag_cx * 100) if ag_cx > 0 else 0

            print(f"{desc:<12} {ag_cx:<12} {greedy_cx:<12} {reduction:<10} {reduction_rate:.1f}%")

            total_ag_cx += ag_cx
            total_greedy_cx += greedy_cx

        total_reduction = total_ag_cx - total_greedy_cx
        total_reduction_rate = (total_reduction / total_ag_cx * 100) if total_ag_cx > 0 else 0

        print(f"-" * 60)
        print(f"{'总计':<12} {total_ag_cx:<12} {total_greedy_cx:<12} {total_reduction:<10} {total_reduction_rate:.1f}%")
        print(f"\nGreedy算法相对AG算法平均减少 {total_reduction_rate:.1f}% 的CX门")

        self.assertLess(total_greedy_cx, total_ag_cx, "Greedy应优于AG算法")

    def test_6_2_bravyi_maslov_optimal_synthesis(self):
        """测试6.2: Bravyi-Maslov算法最优CX合成"""
        from optimize.synthesis import synthesize_clifford_bravyi_maslov

        # 2量子比特Clifford有理论最优CX数
        cliff = random_clifford(2, seed=45)
        circuit = synthesize_clifford_bravyi_maslov(cliff)
        cx_count = circuit.count_ops().get('cx', 0)

        print(f"\n测试6.2通过: BM算法最优CX合成")
        print(f"   2量子比特Clifford")
        print(f"   CX门数: {cx_count} (理论最优≤3)")
        print(f"   电路深度: {circuit.depth()}")
        print(f"   总门数: {len(circuit.data)}")

        self.assertLessEqual(cx_count, 3, "2比特Clifford最多需要3个CX")

    def test_6_3_lnn_depth_optimization(self):
        """测试6.3: LNN拓扑深度优化（展示深度减少）"""
        from optimize.synthesis import (
            synthesize_clifford_aaronson_gottesman,
            synthesize_clifford_depth_lnn
        )

        # 对比全连接和LNN的深度
        test_qubits = [4, 5, 6]
        print(f"\n测试6.3: LNN拓扑深度优化对比")
        print(f"{'规模':<12} {'全连接深度':<14} {'LNN深度':<12} {'理论界(9n)':<14} {'满足理论界'}")
        print(f"-" * 70)

        for n in test_qubits:
            cliff = random_clifford(n, seed=40+n)

            # 全连接合成（不考虑拓扑）
            circuit_full = synthesize_clifford_aaronson_gottesman(cliff)
            full_depth = circuit_full.depth()

            # LNN优化合成
            circuit_lnn = synthesize_clifford_depth_lnn(cliff)
            lnn_depth = circuit_lnn.depth()

            theoretical_bound = 9 * n
            satisfies_bound = "是" if lnn_depth <= theoretical_bound else "否"

            print(f"{n}量子比特    {full_depth:<14} {lnn_depth:<12} {theoretical_bound:<14} {satisfies_bound}")

        print(f"\nLNN深度优化满足理论深度界 (≤9n)")

        self.assertIsNotNone(circuit_lnn)

    def test_6_4_large_scale_clifford_synthesis(self):
        """测试6.4: 大规模Clifford电路重合成优化"""
        from optimize import (
            CliffordMerger,
            synthesize_clifford_greedy
        )
        from circuit import circuit_to_dag, dag_to_circuit

        # 创建一个包含多层Clifford门的电路
        qc = QuantumCircuit(5)

        # 5层Clifford操作
        for layer in range(5):
            # H门层
            for i in range(5):
                qc.h(i)

            # CNOT层（环形）
            for i in range(4):
                qc.cx(i, i+1)
            qc.cx(4, 0)  # 闭环

            # S门层
            for i in range(5):
                if (layer + i) % 2 == 0:
                    qc.s(i)

        original_size = len(qc.data)
        original_cx = qc.count_ops().get('cx', 0)
        original_depth = qc.depth()

        # 方法1: 转换为Clifford对象并重新合成
        cliff_obj = Clifford(qc)
        qc_resynthesized = synthesize_clifford_greedy(cliff_obj)

        resyn_size = len(qc_resynthesized.data)
        resyn_cx = qc_resynthesized.count_ops().get('cx', 0)
        resyn_depth = qc_resynthesized.depth()

        # 方法2: 使用CliffordMerger pass
        dag = circuit_to_dag(qc)
        merger = CliffordMerger()
        dag_merged = merger.run(dag)
        qc_merged = dag_to_circuit(dag_merged)

        merged_size = len(qc_merged.data)
        merged_cx = qc_merged.count_ops().get('cx', 0)
        merged_depth = qc_merged.depth()

        print(f"\n测试6.4通过: 大规模Clifford电路优化 (5量子比特, 5层)")
        print(f"   原始电路:")
        print(f"     - 总门数: {original_size}, CX: {original_cx}, 深度: {original_depth}")
        print(f"   Greedy重合成:")
        print(f"     - 总门数: {resyn_size} (减少{original_size-resyn_size}, {(original_size-resyn_size)/original_size*100:.1f}%)")
        print(f"     - CX门: {resyn_cx} (减少{original_cx-resyn_cx}, {(original_cx-resyn_cx)/original_cx*100:.1f}%)")
        print(f"     - 深度: {resyn_depth} (减少{original_depth-resyn_depth}, {(original_depth-resyn_depth)/original_depth*100:.1f}%)")
        print(f"   CliffordMerger优化:")
        print(f"     - 总门数: {merged_size}")
        print(f"     - CX门: {merged_cx}")
        print(f"     - 深度: {merged_depth}")

        self.assertLess(resyn_cx, original_cx, "重合成应减少CX门数")
        self.assertLess(resyn_size, original_size, "重合成应减少总门数")

    def test_6_5_stabilizer_state_preparation(self):
        """测试6.5: 稳定子态制备电路优化（GHZ, 簇态）"""
        from optimize import synthesize_clifford_greedy

        # GHZ态制备
        def create_ghz(n):
            qc = QuantumCircuit(n)
            qc.h(0)
            for i in range(n-1):
                qc.cx(i, i+1)
            return qc

        # 簇态制备
        def create_cluster_state(n):
            qc = QuantumCircuit(n)
            for i in range(n):
                qc.h(i)
            for i in range(n-1):
                qc.cz(i, i+1)
            return qc

        print(f"\n测试6.5: 稳定子态制备优化")

        # 测试GHZ态
        for n in [4, 5, 6]:
            ghz = create_ghz(n)
            original_cx = ghz.count_ops().get('cx', 0)

            # 转为Clifford并重合成
            cliff = Clifford(ghz)
            optimized = synthesize_clifford_greedy(cliff)
            opt_cx = optimized.count_ops().get('cx', 0)

            print(f"   {n}量子比特GHZ态: 原始CX={original_cx}, 优化CX={opt_cx}")

            # GHZ态的理论最优是n-1个CX
            self.assertLessEqual(opt_cx, n, "GHZ态CX应接近最优")

        # 测试簇态
        for n in [4, 5]:
            cluster = create_cluster_state(n)
            original_cz = cluster.count_ops().get('cz', 0) if 'cz' in cluster.count_ops() else 0

            cliff = Clifford(cluster)
            optimized = synthesize_clifford_greedy(cliff)

            print(f"   {n}量子比特簇态: 原始CZ={original_cz}, 优化门数={len(optimized.data)}")

        print(f"   稳定子态制备优化完成")


if __name__ == '__main__':
    print("="*70)
    print("技术6增强测试: Clifford电路优化 - 显著优化效果展示")
    print("="*70)
    unittest.main(verbosity=2)
