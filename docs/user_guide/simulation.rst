量子模拟
========

Janus 提供完整的量子电路模拟器，支持状态向量模拟、密度矩阵模拟和噪声模拟。

状态向量模拟器
--------------

``StatevectorSimulator`` 使用完整状态向量进行精确模拟，适用于小规模电路（< 25 量子比特）。

**基本用法**

.. code-block:: python

   from janus.circuit import Circuit
   from janus.simulator import StatevectorSimulator

   # 创建电路
   qc = Circuit.from_layers([
       [{'name': 'h', 'qubits': [0], 'params': []}],
       [{'name': 'cx', 'qubits': [0, 1], 'params': []}],
   ], n_qubits=2)

   # 创建模拟器（可设置随机种子保证可重复性）
   sim = StatevectorSimulator(seed=42)

   # 运行模拟，测量 1000 次
   result = sim.run(qc, shots=1000)

   # 查看结果
   print(result.counts)        # {'00': 503, '11': 497}
   print(result.probabilities) # 概率分布

**获取状态向量**

.. code-block:: python

   # 获取最终状态向量
   sv = sim.statevector(qc)

   print(sv)                    # 状态向量表示
   print(sv.probabilities())    # 概率分布
   print(sv.data)               # 原始复数数组


初始状态
~~~~~~~~

默认初始状态为 |0...0⟩，可以指定其他初始状态：

.. code-block:: python

   from janus.simulator import Statevector

   # 方法 1：字符串标签
   result = sim.run(qc, shots=100, initial_state='01')

   # 方法 2：Statevector 对象
   sv_init = Statevector.from_label('+0')  # |+⟩⊗|0⟩
   result = sim.run(qc, shots=100, initial_state=sv_init)

   # 方法 3：自定义数组
   import numpy as np
   custom_state = np.array([1, 0, 0, 1]) / np.sqrt(2)
   result = sim.run(qc, shots=100, initial_state=custom_state)

部分测量
~~~~~~~~

只测量部分量子比特：

.. code-block:: python

   qc = Circuit.from_layers([
       [
           {'name': 'h', 'qubits': [0], 'params': []},
           {'name': 'h', 'qubits': [1], 'params': []},
           {'name': 'h', 'qubits': [2], 'params': []},
       ]
   ], n_qubits=3)

   # 只测量 qubit 0 和 qubit 2
   result = sim.run(qc, shots=1000, measure_qubits=[0, 2])
   print(result.counts)  # 2 位结果

参数化电路模拟
~~~~~~~~~~~~~~

.. code-block:: python

   from janus.circuit import Circuit, Parameter
   import numpy as np

   theta = Parameter('theta')

   qc = Circuit.from_layers([
       [{'name': 'ry', 'qubits': [0], 'params': [theta]}],
   ], n_qubits=1)

   sim = StatevectorSimulator()

   # 绑定参数运行
   result = sim.run(qc, shots=100, parameter_binds={'theta': np.pi})

   # 参数扫描
   for t in [0, np.pi/4, np.pi/2, np.pi]:
       sv = sim.statevector(qc, parameter_binds={'theta': t})
       prob_1 = sv.probabilities()[1]
       print(f"θ = {t:.2f}: P(|1⟩) = {prob_1:.3f}")

期望值计算
~~~~~~~~~~

计算可观测量的期望值：

.. code-block:: python

   import numpy as np
   from janus.circuit import Circuit
   from janus.simulator import StatevectorSimulator

   # Pauli 矩阵
   Z = np.array([[1, 0], [0, -1]])
   X = np.array([[0, 1], [1, 0]])

   qc = Circuit.from_layers([
       [{'name': 'h', 'qubits': [0], 'params': []}],
   ], n_qubits=1)

   sim = StatevectorSimulator()
   sv = sim.statevector(qc)

   print(f"⟨Z⟩ = {sv.expectation_value(Z).real:.3f}")  # ≈ 0
   print(f"⟨X⟩ = {sv.expectation_value(X).real:.3f}")  # = 1

Statevector 类
--------------

``Statevector`` 类提供丰富的状态向量操作：

**创建状态**

.. code-block:: python

   from janus.simulator import Statevector
   import numpy as np

   # 从整数创建（计算基态）
   sv0 = Statevector.from_int(0, num_qubits=3)  # |000⟩
   sv5 = Statevector.from_int(5, num_qubits=3)  # |101⟩

   # 从标签创建
   sv = Statevector.from_label('00')   # |00⟩
   sv = Statevector.from_label('+-')   # |+⟩⊗|-⟩
   sv = Statevector.from_label('01r')  # |0⟩⊗|1⟩⊗|r⟩

   # 从电路创建
   sv = Statevector.from_circuit(qc)

   # 从数组创建
   sv = Statevector(np.array([1, 0, 0, 1]) / np.sqrt(2))

**状态操作**

.. code-block:: python

   sv.probabilities()           # 完整概率分布
   sv.probabilities([0])        # qubit 0 的边缘概率
   sv.sample_counts(1000)       # 采样 1000 次
   sv.expectation_value(op)     # 期望值

   # 状态比较
   sv1.inner(sv2)               # 内积 ⟨sv1|sv2⟩
   sv1.equiv(sv2)               # 是否等价（忽略全局相位）

   # 张量积
   sv_tensor = sv1.tensor(sv2)  # |sv1⟩⊗|sv2⟩

   # 演化
   H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
   sv.evolve(H, [0])            # 在 qubit 0 上应用 H


密度矩阵
--------

``DensityMatrix`` 可以描述混合态，用于噪声模拟：

.. code-block:: python

   from janus.simulator import DensityMatrix, Statevector

   # 从状态向量创建（纯态）
   sv = Statevector.from_circuit(qc)
   dm = DensityMatrix.from_statevector(sv)

   # 密度矩阵属性
   print(dm.purity())              # 纯度（1 = 纯态，< 1 = 混合态）
   print(dm.is_pure())             # 是否纯态
   print(dm.von_neumann_entropy()) # 冯诺依曼熵（纯态 = 0）
   print(dm.trace())               # 迹（应该 = 1）

   # 部分迹（约化密度矩阵）
   dm_reduced = dm.partial_trace([1])  # 对 qubit 1 求迹，保留 qubit 0

   # 保真度
   dm1.fidelity(dm2)  # 两个密度矩阵的保真度

噪声模拟
--------

``NoisySimulator`` 使用密度矩阵模拟噪声效应。

**创建噪声模型**

.. code-block:: python

   from janus.simulator import (
       NoisySimulator,
       NoiseModel,
       depolarizing_channel,
       amplitude_damping_channel,
       phase_damping_channel
   )

   noise_model = NoiseModel()

   # 为单比特门添加噪声
   noise_model.add_all_qubit_quantum_error(
       depolarizing_channel(0.01),  # 1% 去极化噪声
       ['h', 'x', 'rx', 'ry', 'rz']
   )

   # 为两比特门添加噪声
   noise_model.add_all_qubit_quantum_error(
       depolarizing_channel(0.02),  # 2% 去极化噪声
       ['cx', 'cz']
   )

**运行噪声模拟**

.. code-block:: python

   # 创建电路
   qc = Circuit.from_layers([
       [{'name': 'h', 'qubits': [0], 'params': []}],
       [{'name': 'cx', 'qubits': [0, 1], 'params': []}],
   ], n_qubits=2)

   # 噪声模拟
   noisy_sim = NoisySimulator(noise_model, seed=42)
   result = noisy_sim.run(qc, shots=1000)

   print(result.counts)
   # 会包含错误结果，如 {'00': 480, '11': 470, '01': 25, '10': 25}

   # 获取密度矩阵
   dm = noisy_sim.density_matrix(qc)
   print(f"纯度: {dm.purity():.4f}")  # < 1 表示混合态

噪声信道
~~~~~~~~

Janus 提供多种噪声信道：

.. code-block:: python

   from janus.simulator import (
       depolarizing_channel,      # 去极化噪声
       amplitude_damping_channel, # 振幅阻尼 (T1 衰减)
       phase_damping_channel,     # 相位阻尼 (T2 退相干)
       bit_flip_channel,          # 比特翻转
       phase_flip_channel,        # 相位翻转
       thermal_relaxation_channel,# 热弛豫
       readout_error_channel,     # 读出错误
   )

   # 去极化：以概率 p 随机应用 X, Y, Z
   dep = depolarizing_channel(p=0.01)

   # 振幅阻尼：模拟能量衰减
   amp = amplitude_damping_channel(gamma=0.05)

   # 相位阻尼：模拟相位信息丢失
   phase = phase_damping_channel(gamma=0.03)

   # 比特翻转：以概率 p 应用 X 门
   bit_flip = bit_flip_channel(p=0.01)

   # 相位翻转：以概率 p 应用 Z 门
   phase_flip = phase_flip_channel(p=0.01)

**针对特定量子比特的噪声**

.. code-block:: python

   noise_model = NoiseModel()

   # 只在 qubit 0 上添加噪声
   noise_model.add_quantum_error(
       depolarizing_channel(0.05),
       ['h', 'x'],
       [0]  # 只影响 qubit 0
   )

模拟结果
--------

``SimulatorResult`` 包含模拟结果：

.. code-block:: python

   result = sim.run(qc, shots=1000)

   # 测量统计
   result.counts           # {'00': 503, '11': 497}
   result.counts.most_frequent()  # '00' 或 '11'

   # 概率
   result.probabilities    # 概率分布

   # 状态向量（如果 return_statevector=True）
   result.statevector      # 最终状态向量

   # 元数据
   result.shots            # 测量次数
   result.metadata         # 其他信息

性能提示
--------

1. **量子比特数限制**: 状态向量模拟的内存需求为 O(2^n)，建议 n < 25

2. **使用随机种子**: 设置 ``seed`` 参数保证结果可重复

3. **批量运行**: 使用 ``run_batch`` 批量运行多个电路

.. code-block:: python

   circuits = [create_circuit(i) for i in range(10)]
   results = sim.run_batch(circuits, shots=1000)

4. **精度选择**: 使用 ``precision='single'`` 可以减少内存使用

.. code-block:: python

   sim = StatevectorSimulator(precision='single')
