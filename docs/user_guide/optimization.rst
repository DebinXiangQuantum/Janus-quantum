电路优化
========

Janus 提供完整的量子电路优化框架，包含 10 种优化技术。

快速开始
--------

**智能优化（推荐）**

.. code-block:: python

   from janus.optimize import smart_optimize

   # 创建电路
   qc = Circuit.from_layers([
       [{'name': 'h', 'qubits': [0], 'params': []}],
       [{'name': 'h', 'qubits': [0], 'params': []}],  # 冗余
       [{'name': 't', 'qubits': [0], 'params': []}],
       [{'name': 't', 'qubits': [0], 'params': []}],  # 可合并
   ], n_qubits=2)

   # 智能优化（自动选择最优策略）
   qc_opt = smart_optimize(qc, level=2, verbose=True)

**基本编译**

.. code-block:: python

   from janus.compiler import compile_circuit

   optimized = compile_circuit(qc, optimization_level=2)

优化级别
--------

.. list-table::
   :header-rows: 1
   :widths: 10 90

   * - 级别
     - 内容
   * - 0
     - 无优化
   * - 1
     - 基础优化（逆门消除）
   * - 2
     - 标准优化（T门合并 + Clifford合并 + 交换性消除）
   * - 3
     - 完整优化（所有技术）

十大优化技术
------------

Janus 提供统一 API 访问所有优化技术：

.. code-block:: python

   from janus.optimize import (
       optimize_clifford_rz,      # 技术1: Clifford+RZ优化
       optimize_gate_fusion,      # 技术2: 门融合优化
       optimize_commutativity,    # 技术3: 交换性优化
       optimize_template,         # 技术4: 模板匹配优化
       optimize_kak,              # 技术5: KAK分解优化
       optimize_clifford_synth,   # 技术6: Clifford合成优化
       optimize_cnot_synth,       # 技术7: CNOT合成优化
       run_benchmark,             # 技术8: 基准测试
       analyze_circuit,           # 技术9: 电路分析
       smart_optimize,            # 技术10: 智能优化
   )

技术1: Clifford+RZ 优化
~~~~~~~~~~~~~~~~~~~~~~~

将电路分解为 Clifford 门和 RZ 旋转门，优化 T 门数量。

- T门合并规则: T+T→S, T+T+T+T→Z
- 适用于容错量子计算电路

.. code-block:: python

   from janus.optimize import optimize_clifford_rz

   qc_opt = optimize_clifford_rz(
       qc,
       enable_t_merge=True,        # T门合并
       enable_clifford_merge=True, # Clifford门合并
       enable_inverse_cancel=True, # 逆门消除
       verbose=True
   )

技术2: 门融合优化
~~~~~~~~~~~~~~~~~

将连续的单量子比特门融合为一个等价门。

.. code-block:: python

   from janus.optimize import optimize_gate_fusion

   qc_opt = optimize_gate_fusion(
       qc,
       enable_rotation_merge=True,     # 旋转门合并
       enable_single_qubit_opt=True,   # 单比特门优化
       enable_block_consolidate=False, # 块合并（较慢）
       verbose=True
   )

技术3: 交换性优化
~~~~~~~~~~~~~~~~~

利用量子门的交换性重新排列门顺序，消除冗余门。

- 自伴门消除: H·H=I, X·X=I, CX·CX=I
- 互逆门消除: T·Tdg=I, S·Sdg=I

.. code-block:: python

   from janus.optimize import optimize_commutativity

   qc_opt = optimize_commutativity(
       qc,
       enable_commutative_cancel=True,   # 交换性消除
       enable_inverse_cancel=True,       # 逆门消除
       enable_commutative_inverse=True,  # 交换性逆门消除
       verbose=True
   )

技术4: 模板匹配优化
~~~~~~~~~~~~~~~~~~~

识别电路中的已知模式，用更优的等价电路替换。

.. code-block:: python

   from janus.optimize import optimize_template

   qc_opt = optimize_template(
       qc,
       enable_template_match=True,      # 模板匹配
       enable_inverse_cancel=True,      # 逆门消除
       template_list=None,              # 自定义模板（可选）
       verbose=True
   )

.. note::
   模板匹配在大电路（>100门）上较慢，会自动跳过。

技术5: KAK 分解优化
~~~~~~~~~~~~~~~~~~~

使用 Khaneja-Glaser (KAK) 分解优化任意双量子比特门。

.. code-block:: python

   from janus.optimize import optimize_kak

   qc_opt = optimize_kak(
       qc,
       enable_block_collect=True,      # 两比特块收集
       enable_block_consolidate=True,  # 块合并
       basis_gate='cx',                # 基础门: 'cx', 'cz', 'iswap'
       verbose=True
   )

技术6: Clifford 合成优化
~~~~~~~~~~~~~~~~~~~~~~~~

优化 Clifford 门电路的合成。

.. code-block:: python

   from janus.optimize import optimize_clifford_synth

   qc_opt = optimize_clifford_synth(
       qc,
       method='greedy',  # 'greedy', 'bravyi_maslov', 'ag', 'depth_lnn'
       verbose=True
   )

技术7: CNOT 合成优化
~~~~~~~~~~~~~~~~~~~~

优化 CNOT 门网络，减少 CNOT 门数量。

.. code-block:: python

   from janus.optimize import optimize_cnot_synth

   qc_opt = optimize_cnot_synth(
       qc,
       method='pmh',  # 'pmh', 'lnn_kms', 'phase_aam'
       verbose=True
   )

技术8: 基准测试
~~~~~~~~~~~~~~~

对电路进行多级别优化测试，评估优化效果。

.. code-block:: python

   from janus.optimize import run_benchmark

   results = run_benchmark(
       qc,
       optimization_levels=[0, 1, 2, 3],
       verbose=True
   )

   # 查看结果
   for level, stats in results['levels'].items():
       print(f"Level {level}: {stats['size']}门, 减少{stats['reduction']:.1f}%")

技术9: 电路分析
~~~~~~~~~~~~~~~

收集和分析电路的详细指标。

.. code-block:: python

   from janus.optimize import analyze_circuit

   metrics = analyze_circuit(qc, verbose=True)

   print(f"门数: {metrics['size']}")
   print(f"深度: {metrics['depth']}")
   print(f"宽度: {metrics['width']}")
   print(f"门统计: {metrics['ops']}")
   print(f"单比特门: {metrics['n_single_qubit']}")
   print(f"两比特门: {metrics['n_two_qubit']}")

技术10: 智能优化
~~~~~~~~~~~~~~~~

自动分析电路特征，智能选择最优的优化技术组合。

.. code-block:: python

   from janus.optimize import smart_optimize

   # 自动检测电路类型并优化
   qc_opt = smart_optimize(qc, level=2, verbose=True)

   # 强制使用特定策略
   qc_opt = smart_optimize(
       qc,
       strategy='t_heavy',  # 't_heavy', 'rotation_heavy', 'clifford_heavy', 'cx_heavy', 'mixed'
       verbose=True
   )

**SmartOptimizer 类**

.. code-block:: python

   from janus.optimize import SmartOptimizer, analyze_and_optimize

   # 使用优化器类
   optimizer = SmartOptimizer(verbose=True)
   qc_opt = optimizer.optimize(qc)

   # 带详细报告的优化
   report = analyze_and_optimize(qc, verbose=True)
   print(f"策略: {report['strategy']}")
   print(f"门数减少: {report['improvements']['gate_reduction']:.1f}%")
   print(f"深度减少: {report['improvements']['depth_reduction']:.1f}%")

底层 Pass 类
------------

如需更精细的控制，可以直接使用底层 Pass 类：

**Clifford+RZ 优化 Pass**

.. code-block:: python

   from janus.optimize import (
       TChinMerger,           # T门合并
       CliffordMerger,        # Clifford门合并
       CollectCliffords,      # Clifford门收集
       LitinskiTransformation # Litinski变换
   )

**门融合 Pass**

.. code-block:: python

   from janus.optimize import (
       ConsolidateBlocks,     # 块合并
       Optimize1qGates,       # 单比特门优化
       Collect1qRuns,         # 单比特门序列收集
       Collect2qBlocks,       # 两比特块收集
       Split2QUnitaries       # 两比特酉矩阵分解
   )

**交换性优化 Pass**

.. code-block:: python

   from janus.optimize import (
       CommutativeCancellation,        # 交换性消除
       InverseCancellation,            # 逆门消除
       CommutativeInverseCancellation, # 交换性逆门消除
       CommutationAnalysis             # 交换性分析
   )

**模板匹配 Pass**

.. code-block:: python

   from janus.optimize import (
       TemplateOptimization,   # 模板优化
       TemplateMatching,       # 模板匹配
       TemplateSubstitution    # 模板替换
   )

**分析 Pass**

.. code-block:: python

   from janus.optimize import (
       Depth,              # 深度分析
       Width,              # 宽度分析
       Size,               # 大小分析
       CountOps,           # 门统计
       ResourceEstimation  # 资源估算
   )

合成算法
--------

**KAK 分解**

.. code-block:: python

   from janus.optimize import (
       TwoQubitWeylDecomposition,  # Weyl分解
       TwoQubitBasisDecomposer,    # 基础门分解
       two_qubit_cnot_decompose    # CNOT分解
   )

**Clifford 合成**

.. code-block:: python

   from janus.optimize import (
       synthesize_clifford_circuit,
       synthesize_clifford_greedy,
       synthesize_clifford_bravyi_maslov,
       synthesize_clifford_depth_lnn
   )

**CNOT 合成**

.. code-block:: python

   from janus.optimize import (
       synthesize_cnot_count_pmh,
       synthesize_cnot_depth_lnn_kms,
       synthesize_cnot_phase_aam
   )

自定义优化流程
--------------

.. code-block:: python

   from janus.circuit import circuit_to_dag, dag_to_circuit
   from janus.optimize import (
       TChinMerger, CliffordMerger,
       InverseCancellation, CommutativeCancellation
   )

   # 转换为 DAG
   dag = circuit_to_dag(qc)

   # 依次应用优化 Pass
   dag = TChinMerger().run(dag)
   dag = CliffordMerger().run(dag)
   dag = InverseCancellation().run(dag)
   dag = CommutativeCancellation().run(dag)

   # 转回电路
   qc_opt = dag_to_circuit(dag)

完整示例
--------

.. code-block:: python

   from janus.circuit import Circuit
   from janus.optimize import smart_optimize, analyze_circuit
   import numpy as np

   # 创建一个有冗余的电路
   qc = Circuit.from_layers([
       [{'name': 'h', 'qubits': [0], 'params': []}],
       [{'name': 'h', 'qubits': [0], 'params': []}],      # H·H = I
       [{'name': 't', 'qubits': [0], 'params': []}],
       [{'name': 't', 'qubits': [0], 'params': []}],      # T·T = S
       [{'name': 'cx', 'qubits': [0, 1], 'params': []}],
       [{'name': 'cx', 'qubits': [0, 1], 'params': []}],  # CX·CX = I
       [{'name': 'rz', 'qubits': [1], 'params': [0.5]}],
       [{'name': 'rz', 'qubits': [1], 'params': [0.5]}],  # 可合并
   ], n_qubits=2)

   print("=== 优化前 ===")
   before = analyze_circuit(qc)
   print(f"门数: {before['size']}, 深度: {before['depth']}")

   # 智能优化
   qc_opt = smart_optimize(qc, level=2, verbose=True)

   print("\n=== 优化后 ===")
   after = analyze_circuit(qc_opt)
   print(f"门数: {after['size']}, 深度: {after['depth']}")
   print(f"减少: {before['size'] - after['size']} 门")
