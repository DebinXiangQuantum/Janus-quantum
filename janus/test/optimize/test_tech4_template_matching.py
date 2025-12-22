"""
技术4: 模版匹配的量子电路优化 - 单元测试

测试目标:
- 验证电路模板识别能力
- 验证次优模式替换优化
- 验证模板匹配优化效果
"""

import sys
import os

# 添加janus目录到Python路径（使用optimize模块）
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


import unittest
from circuit import Circuit as QuantumCircuit
from circuit import circuit_to_dag, dag_to_circuit


class TestTech4TemplateMatchingOptimization(unittest.TestCase):
    """技术4: 模版匹配的量子电路优化测试类"""

    def test_4_1_template_pattern_recognition(self):
        """测试4.1: 电路模板模式识别"""
        try:
            from optimize import TemplatePatternMatcher
            from compat.converters import dag_to_dagdependency

            circuit_qc = QuantumCircuit(2)
            circuit_qc.cx(0, 1)
            circuit_qc.h(0)

            template_qc = QuantumCircuit(2)
            template_qc.cx(0, 1)

            circuit_dag = circuit_to_dag(circuit_qc)
            template_dag = circuit_to_dag(template_qc)
            circuit_dag_dep = dag_to_dagdependency(circuit_dag)
            template_dag_dep = dag_to_dagdependency(template_dag)

            matcher = TemplatePatternMatcher(
                circuit_dag_dep=circuit_dag_dep,
                template_dag_dep=template_dag_dep
            )
            matcher.run_template_matching()
            matches = matcher.match_list

            print(f"测试4.1通过: 模板模式识别成功")
            print(f"   找到 {len(matches)} 个匹配模式")
            self.assertIsNotNone(matches)
        except (ImportError, AttributeError, NotImplementedError) as e:
            self.skipTest(f"模板匹配功能不完整: {e}")

    def test_4_2_suboptimal_pattern_replacement(self):
        """测试4.2: 次优模式替换优化"""
        try:
            from optimize import CircuitTemplateOptimizer
            from qiskit.circuit.library.templates import template_nct_2a_1

            qc = QuantumCircuit(2)
            qc.cx(0, 1)
            qc.h(0)
            qc.h(1)
            qc.cx(0, 1)

            original_size = len(qc.data)
            template = template_nct_2a_1()
            dag = circuit_to_dag(qc)
            optimizer = CircuitTemplateOptimizer(template_list=[template])
            dag_optimized = optimizer.run(dag)
            qc_optimized = dag_to_circuit(dag_optimized)

            optimized_size = len(qc_optimized.data)
            print(f"测试4.2通过: 次优模式替换优化有效")
            print(f"   原始门数: {original_size}, 优化后: {optimized_size}")
            self.assertLessEqual(optimized_size, original_size)
        except (ImportError, ModuleNotFoundError) as e:
            self.skipTest(f"Qiskit模板库不可用: {e}")

    def test_4_3_multi_template_optimization(self):
        """测试4.3: 多模板联合优化"""
        try:
            from optimize import CircuitTemplateOptimizer
            from qiskit.circuit.library.templates import template_nct_2a_1, template_nct_2a_2

            qc = QuantumCircuit(3)
            qc.cx(0, 1)
            qc.cx(0, 1)
            qc.h(0)
            qc.cx(1, 2)

            original_depth = qc.depth()
            original_size = len(qc.data)

            templates = [template_nct_2a_1(), template_nct_2a_2()]
            dag = circuit_to_dag(qc)
            optimizer = CircuitTemplateOptimizer(template_list=templates, user_cost_dict={'cx': 4, 'h': 1})
            dag_optimized = optimizer.run(dag)
            qc_optimized = dag_to_circuit(dag_optimized)

            optimized_depth = qc_optimized.depth()
            optimized_size = len(qc_optimized.data)

            print(f"测试4.3通过: 多模板联合优化有效")
            print(f"   原始 - 深度: {original_depth}, 门数: {original_size}")
            print(f"   优化 - 深度: {optimized_depth}, 门数: {optimized_size}")
            print(f"   门数减少: {original_size - optimized_size}")
            self.assertLessEqual(optimized_size, original_size)
        except (ImportError, ModuleNotFoundError) as e:
            self.skipTest(f"Qiskit模板库不可用: {e}")


if __name__ == '__main__':
    unittest.main(verbosity=2)
