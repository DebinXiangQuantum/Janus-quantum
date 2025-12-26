核心概念
========

本节介绍 Janus 的核心概念和设计理念，帮助你更好地理解和使用框架。

量子比特 (Qubit)
----------------

量子比特是量子计算的基本信息单元，与经典比特不同，它可以处于 |0⟩ 和 |1⟩ 的叠加态。

.. code-block:: python

   from janus.circuit import Circuit

   # 创建 3 个量子比特的电路，所有量子比特初始状态为 |0⟩
   # 在同一层的门可以并行执行
   qc = Circuit.from_layers([
       [
           {'name': 'h', 'qubits': [0], 'params': []},   # 操作 qubit 0
           {'name': 'x', 'qubits': [1], 'params': []},   # 操作 qubit 1
           {'name': 'z', 'qubits': [2], 'params': []},   # 操作 qubit 2
       ]
   ], n_qubits=3)

**量子比特编号约定**

在 Janus 中，量子比特从 0 开始编号。测量结果字符串中，最右边的位对应 qubit 0：

.. code-block:: python

   # 测量结果 '10' 表示:
   # qubit 1 = 1, qubit 0 = 0

量子门 (Gate)
-------------

量子门是作用在量子比特上的酉变换，是量子计算的基本操作。

**单比特门**

作用在单个量子比特上：

.. code-block:: python

   from janus.circuit import Circuit

   qc = Circuit.from_layers([
       [{'name': 'h', 'qubits': [0], 'params': []}],      # Hadamard 门：创建叠加态
       [{'name': 'x', 'qubits': [0], 'params': []}],      # Pauli-X 门：比特翻转
       [{'name': 'z', 'qubits': [0], 'params': []}],      # Pauli-Z 门：相位翻转
       [{'name': 'rx', 'qubits': [0], 'params': [1.57]}], # 绕 X 轴旋转 π/2
   ], n_qubits=1)

**两比特门**

作用在两个量子比特上，通常一个是控制比特，一个是目标比特：

.. code-block:: python

   qc = Circuit.from_layers([
       [{'name': 'cx', 'qubits': [0, 1], 'params': []}],   # CNOT：qubit 0 控制，qubit 1 目标
       [{'name': 'cz', 'qubits': [0, 1], 'params': []}],   # CZ：受控 Z 门
       [{'name': 'swap', 'qubits': [0, 1], 'params': []}], # SWAP：交换两个量子比特的状态
   ], n_qubits=2)

**多比特门**

作用在三个或更多量子比特上：

.. code-block:: python

   qc = Circuit.from_layers([
       [{'name': 'ccx', 'qubits': [0, 1, 2], 'params': []}],  # Toffoli 门：双控制 X
       [{'name': 'mcx', 'qubits': [0, 1, 2], 'params': []}],  # 多控制 X 门
   ], n_qubits=3)

量子电路 (Circuit)
------------------

量子电路是量子门的有序集合，描述了量子计算的完整过程。

**电路结构**

Janus 使用分层结构组织电路，同一层的门可以并行执行：

.. code-block:: python

   from janus.circuit import Circuit

   # 使用 from_layers 创建电路，明确指定每层的门
   qc = Circuit.from_layers([
       [  # 第 0 层：三个门并行执行
           {'name': 'h', 'qubits': [0], 'params': []},
           {'name': 'x', 'qubits': [1], 'params': []},
           {'name': 'z', 'qubits': [2], 'params': []},
       ],
       [  # 第 1 层
           {'name': 'cx', 'qubits': [0, 1], 'params': []},
       ],
   ], n_qubits=3)

**电路属性**

.. code-block:: python

   qc.n_qubits            # 量子比特数
   qc.depth               # 电路深度（层数）
   qc.n_gates             # 门总数
   qc.num_two_qubit_gate  # 两比特门数量
   qc.gates               # 门列表
   qc.layers              # 分层表示

状态向量 (Statevector)
----------------------

状态向量是量子系统的数学描述，包含所有可能测量结果的概率幅。

对于 n 个量子比特，状态向量有 2^n 个复数分量：

.. code-block:: python

   from janus.simulator import Statevector

   # 创建 |00⟩ 状态
   sv = Statevector.from_label('00')
   print(sv.data)  # [1, 0, 0, 0]

   # 创建 |+⟩ 状态（叠加态）
   sv_plus = Statevector.from_label('+')
   print(sv_plus.data)  # [0.707, 0.707]

   # 从电路获取状态向量
   sv = Statevector.from_circuit(qc)

**概率计算**

测量结果的概率是概率幅的模方：

.. code-block:: python

   probs = sv.probabilities()
   # probs[i] = |amplitude[i]|^2

密度矩阵 (DensityMatrix)
------------------------

密度矩阵可以描述混合态（统计混合），用于噪声模拟：

.. code-block:: python

   from janus.simulator import DensityMatrix, Statevector

   # 从状态向量创建密度矩阵
   sv = Statevector.from_label('00')
   dm = DensityMatrix.from_statevector(sv)

   # 密度矩阵属性
   print(dm.purity())              # 纯度（1 表示纯态）
   print(dm.is_pure())             # 是否纯态
   print(dm.von_neumann_entropy()) # 冯诺依曼熵

参数化电路 (Parameterized Circuit)
----------------------------------

参数化电路使用符号参数代替具体数值，可在运行时绑定：

.. code-block:: python

   from janus.circuit import Circuit, Parameter

   # 创建参数
   theta = Parameter('theta')

   # 创建参数化电路
   qc = Circuit.from_layers([
       [{'name': 'ry', 'qubits': [0], 'params': [theta]}],
   ], n_qubits=1)

   # 检查参数
   print(qc.parameters)          # {Parameter(theta)}
   print(qc.is_parameterized())  # True

   # 绑定参数
   bound_qc = qc.bind_parameters({theta: 1.57})

**参数表达式**

参数支持数学运算：

.. code-block:: python

   from janus.circuit import Parameter

   theta = Parameter('theta')
   phi = Parameter('phi')

   # 参数表达式
   expr = 2 * theta + phi
   expr = theta / 2

DAG 表示
--------

有向无环图 (DAG) 是电路的另一种表示，便于分析和优化：

.. code-block:: python

   from janus.circuit import circuit_to_dag, dag_to_circuit

   # 电路转 DAG
   dag = circuit_to_dag(qc)

   # DAG 属性
   print(dag.depth())      # 深度
   print(dag.count_ops())  # 门统计

   # 遍历节点
   for node in dag.op_nodes():
       print(node.name, node.qubits)

   # DAG 转回电路
   qc2 = dag_to_circuit(dag)

噪声模型 (NoiseModel)
---------------------

噪声模型描述量子计算机中的各种错误：

.. code-block:: python

   from janus.simulator import NoiseModel, depolarizing_channel

   noise_model = NoiseModel()

   # 去极化噪声：以概率 p 随机应用 X, Y, Z 门
   noise_model.add_all_qubit_quantum_error(
       depolarizing_channel(0.01),  # 1% 错误率
       ['h', 'x', 'rx']             # 应用到这些门
   )

**常见噪声类型**

- **去极化噪声**: 随机 Pauli 错误
- **振幅阻尼**: 能量衰减 (T1)
- **相位阻尼**: 相位丢失 (T2)
- **比特翻转**: 随机 X 错误
- **相位翻转**: 随机 Z 错误

电路优化
--------

电路优化通过变换减少门数量或深度：

.. code-block:: python

   from janus.compiler import compile_circuit

   # 优化级别
   # 0: 无优化
   # 1: 基础优化（消除恒等门、逆门对）
   # 2: 高级优化（合并旋转门）

   optimized = compile_circuit(qc, optimization_level=2)

**优化 Pass**

优化由多个 Pass 组成，每个 Pass 执行特定变换：

.. code-block:: python

   from janus.compiler.passes import (
       RemoveIdentityPass,
       CancelInversesPass,
       MergeRotationsPass
   )

   # 自定义优化流程
   optimized = compile_circuit(qc, passes=[
       RemoveIdentityPass(),
       CancelInversesPass(),
       MergeRotationsPass(),
   ])

术语表
------

.. glossary::

   量子比特 (Qubit)
      量子计算的基本信息单元，可以处于 |0⟩ 和 |1⟩ 的叠加态。

   量子门 (Gate)
      作用在量子比特上的酉变换。

   量子电路 (Circuit)
      量子门的有序集合。

   状态向量 (Statevector)
      描述量子系统状态的复数向量。

   密度矩阵 (Density Matrix)
      可以描述混合态的量子态表示。

   纠缠 (Entanglement)
      多个量子比特之间的量子关联。

   测量 (Measurement)
      将量子态坍缩为经典结果的操作。

   叠加 (Superposition)
      量子比特同时处于多个状态的能力。
