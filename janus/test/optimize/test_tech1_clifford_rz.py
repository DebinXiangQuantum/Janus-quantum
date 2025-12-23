"""
技术1: Clifford+Rz指令集优化 - 增强测试
目标：展示更显著的T门合并和Clifford优化效果
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import unittest
from circuit import Circuit as QuantumCircuit
from circuit import circuit_to_dag, dag_to_circuit


class TestTech1Enhanced(unittest.TestCase):
    """技术1增强测试：更显著的优化效果"""

    def test_1_enhanced_massive_t_gate_merging(self):
        """增强测试1: 大量T门合并（T+T→S, T+T+T+T→Z）"""
        from optimize import TChinMerger

        # 创建包含大量可合并T门的电路
        qc = QuantumCircuit(4)

        # 第一个量子比特：8个T门 → 应合并为Z + Z = I（消除）
        for _ in range(8):
            qc.t(0)

        # 第二个量子比特：4个T门 → 应合并为Z
        for _ in range(4):
            qc.t(1)

        # 第三个量子比特：2个T门 → 应合并为S
        qc.t(2)
        qc.t(2)

        # 第四个量子比特：6个T门 → 应合并为Z + S
        for _ in range(6):
            qc.t(3)

        # 插入Clifford门以分隔（但不应阻止T门合并）
        qc.h(0)
        qc.cx(0, 1)

        # 更多T门序列
        for _ in range(4):
            qc.t(0)
        qc.t(1)
        qc.t(1)

        original_size = len(qc.data)
        original_ops = qc.count_ops()
        t_gates_original = original_ops.get('t', 0)

        # 应用优化
        dag = circuit_to_dag(qc)
        optimizer = TChinMerger()
        dag_optimized = optimizer.run(dag)
        qc_optimized = dag_to_circuit(dag_optimized)

        optimized_size = len(qc_optimized.data)
        optimized_ops = qc_optimized.count_ops()
        t_gates_optimized = optimized_ops.get('t', 0)

        reduction = original_size - optimized_size
        t_reduction = t_gates_original - t_gates_optimized

        print(f"\n测试1 - 大量T门合并优化:")
        print(f"  原始电路: {original_size}个门 (其中{t_gates_original}个T门)")
        print(f"  优化电路: {optimized_size}个门 (其中{t_gates_optimized}个T门)")
        print(f"  总门数减少: {reduction} ({reduction/original_size*100:.1f}%)")
        print(f"  T门减少: {t_reduction} ({t_reduction/t_gates_original*100:.1f}%)")

        self.assertLess(optimized_size, original_size, "优化应显著减少门数")
        self.assertLess(t_gates_optimized, t_gates_original, "T门数应大幅减少")

    def test_2_enhanced_clifford_commutation(self):
        """增强测试2: 复杂Clifford门交换和合并"""
        from optimize import TChinMerger, CliffordMerger

        # 创建复杂的Clifford + T电路
        qc = QuantumCircuit(3)

        # 模拟量子傅里叶变换风格的电路（大量旋转门）
        for i in range(3):
            qc.h(i)
            qc.t(i)
            qc.h(i)  # H-T-H可以优化

        # CX梯子结构
        for i in range(2):
            qc.cx(i, i+1)
            qc.t(i)
            qc.t(i+1)
            qc.cx(i, i+1)

        # 更多Clifford门
        for i in range(3):
            qc.s(i)
            qc.h(i)
            qc.s(i)
            qc.h(i)  # S-H-S-H序列可以优化

        # 添加可合并的T门
        for i in range(3):
            qc.t(i)
            qc.t(i)  # T+T = S

        original_size = len(qc.data)
        original_depth = qc.depth()

        # 应用多个优化passes
        dag = circuit_to_dag(qc)
        dag = TChinMerger().run(dag)
        dag = CliffordMerger().run(dag)
        qc_optimized = dag_to_circuit(dag)

        optimized_size = len(qc_optimized.data)
        optimized_depth = qc_optimized.depth()

        print(f"\n测试2 - 复杂Clifford+T优化:")
        print(f"  原始: {original_size}门, 深度{original_depth}")
        print(f"  优化: {optimized_size}门, 深度{optimized_depth}")
        print(f"  门数减少: {original_size - optimized_size} ({(original_size-optimized_size)/original_size*100:.1f}%)")
        print(f"  深度减少: {original_depth - optimized_depth} ({(original_depth-optimized_depth)/original_depth*100:.1f}%)")

        self.assertLess(optimized_size, original_size)
        self.assertLessEqual(optimized_depth, original_depth)

    def test_3_enhanced_realistic_algorithm(self):
        """增强测试3: 模拟真实量子算法（Quantum Phase Estimation风格）"""
        from optimize import TChinMerger, CliffordMerger

        n_qubits = 5
        qc = QuantumCircuit(n_qubits)

        # 初始化：Hadamard层
        for i in range(n_qubits):
            qc.h(i)

        # 模拟受控旋转层（产生大量T门）
        for control in range(n_qubits - 1):
            for target in range(control + 1, n_qubits):
                # 受控旋转分解为T门序列
                qc.t(control)
                qc.t(target)
                qc.cx(control, target)
                qc.tdg(target)
                qc.cx(control, target)
                qc.t(target)

        # 再来一层可以合并的操作
        for i in range(n_qubits):
            qc.t(i)
            qc.h(i)
            qc.t(i)
            qc.h(i)

        # 反向QFT风格操作
        for i in range(n_qubits):
            qc.h(i)
            for j in range(i):
                qc.t(i)
                qc.t(i)  # 可合并

        original_size = len(qc.data)
        original_depth = qc.depth()
        original_ops = qc.count_ops()

        # 完整优化流水线
        dag = circuit_to_dag(qc)
        dag = TChinMerger().run(dag)
        dag = CliffordMerger().run(dag)
        qc_optimized = dag_to_circuit(dag)

        optimized_size = len(qc_optimized.data)
        optimized_depth = qc_optimized.depth()
        optimized_ops = qc_optimized.count_ops()

        print(f"\n测试3 - 真实量子算法优化 (QPE风格, {n_qubits}量子比特):")
        print(f"  原始电路: {original_size}门, 深度{original_depth}")
        print(f"    T门: {original_ops.get('t', 0)}, Tdg门: {original_ops.get('tdg', 0)}")
        print(f"  优化电路: {optimized_size}门, 深度{optimized_depth}")
        print(f"    T门: {optimized_ops.get('t', 0)}, Tdg门: {optimized_ops.get('tdg', 0)}")
        print(f"  总优化: 减少{original_size - optimized_size}门 ({(original_size-optimized_size)/original_size*100:.1f}%)")
        print(f"  深度优化: 减少{original_depth - optimized_depth} ({(original_depth-optimized_depth)/original_depth*100:.1f}%)")

        self.assertLess(optimized_size, original_size * 0.85, "应至少减少15%的门")
        self.assertLess(optimized_depth, original_depth * 0.9, "深度应有显著降低")


if __name__ == '__main__':
    print("="*70)
    print("技术1增强测试：Clifford+Rz优化 - 显著优化效果展示")
    print("="*70)
    unittest.main(verbosity=2)
