实用示例
========

本节提供一系列完整的量子计算示例，帮助你理解如何使用 Janus 解决实际问题。

Bell 态和量子纠缠
-----------------

Bell 态是最简单的量子纠缠态，由两个量子比特组成：

.. code-block:: python

   from janus.circuit import Circuit
   from janus.simulator import StatevectorSimulator

   def create_bell_state(bell_type='00'):
       """
       创建四种 Bell 态之一

       Args:
           bell_type: '00' (Φ+), '01' (Φ-), '10' (Ψ+), '11' (Ψ-)

       Returns:
           Circuit: Bell 态电路
       """
       layers = []

       # 第一层：H 门和可选的 X 门
       layer0 = [{'name': 'h', 'qubits': [0], 'params': []}]
       if bell_type[0] == '1':
           layer0.append({'name': 'x', 'qubits': [0], 'params': []})
       if bell_type[1] == '1':
           layer0.append({'name': 'x', 'qubits': [1], 'params': []})
       layers.append(layer0)

       # 第二层：CNOT 门
       layers.append([{'name': 'cx', 'qubits': [0, 1], 'params': []}])

       return Circuit.from_layers(layers, n_qubits=2)

   # 创建并模拟 Bell 态
   sim = StatevectorSimulator(seed=42)

   for bell_type in ['00', '01', '10', '11']:
       qc = create_bell_state(bell_type)
       result = sim.run(qc, shots=1000)
       print(f"Bell 态 {bell_type}: {result.counts}")

GHZ 态
------

GHZ 态是多量子比特纠缠态的推广：

.. code-block:: python

   from janus.circuit import Circuit
   from janus.simulator import StatevectorSimulator

   def create_ghz_state(n_qubits):
       """
       创建 n 量子比特的 GHZ 态

       GHZ 态: (|00...0⟩ + |11...1⟩) / √2
       """
       layers = []

       # 第一层：H 门
       layers.append([{'name': 'h', 'qubits': [0], 'params': []}])

       # 后续层：级联 CNOT 门
       for i in range(n_qubits - 1):
           layers.append([{'name': 'cx', 'qubits': [i, i + 1], 'params': []}])

       return Circuit.from_layers(layers, n_qubits=n_qubits)

   # 创建 5 量子比特 GHZ 态
   qc = create_ghz_state(5)
   print(qc.draw())

   sim = StatevectorSimulator()
   result = sim.run(qc, shots=1000)
   print(f"GHZ 态测量结果: {result.counts}")
   # 应该只有 '00000' 和 '11111'


量子隐形传态
------------

量子隐形传态是量子通信的基础协议：

.. code-block:: python

   from janus.circuit import Circuit
   from janus.simulator import StatevectorSimulator
   import numpy as np

   def quantum_teleportation(theta, phi):
       """
       量子隐形传态协议

       将 qubit 0 的状态传送到 qubit 2
       """
       layers = [
           # 准备要传送的状态（在 qubit 0 上）
           [
               {'name': 'ry', 'qubits': [0], 'params': [theta]},
               {'name': 'rz', 'qubits': [0], 'params': [phi]},
           ],
           # 创建 Bell 对（qubit 1 和 qubit 2）
           [{'name': 'h', 'qubits': [1], 'params': []}],
           [{'name': 'cx', 'qubits': [1, 2], 'params': []}],
           # Alice 的操作
           [{'name': 'cx', 'qubits': [0, 1], 'params': []}],
           [{'name': 'h', 'qubits': [0], 'params': []}],
       ]

       return Circuit.from_layers(layers, n_qubits=3)

   # 传送状态 |+⟩
   qc = quantum_teleportation(np.pi/2, 0)
   print("量子隐形传态电路:")
   print(qc.draw())

Grover 搜索算法
---------------

Grover 算法可以在无序数据库中进行二次加速搜索：

.. code-block:: python

   from janus.circuit import Circuit
   from janus.simulator import StatevectorSimulator
   import numpy as np

   def grover_search_2qubit(marked_state='11'):
       """
       2 量子比特 Grover 搜索

       Args:
           marked_state: 目标状态（如 '11'）
       """
       layers = []

       # 初始化：均匀叠加
       layers.append([
           {'name': 'h', 'qubits': [0], 'params': []},
           {'name': 'h', 'qubits': [1], 'params': []},
       ])

       # Oracle：标记目标状态
       oracle_layer = []
       if marked_state[1] == '0':  # qubit 0
           oracle_layer.append({'name': 'x', 'qubits': [0], 'params': []})
       if marked_state[0] == '0':  # qubit 1
           oracle_layer.append({'name': 'x', 'qubits': [1], 'params': []})
       if oracle_layer:
           layers.append(oracle_layer)

       layers.append([{'name': 'cz', 'qubits': [0, 1], 'params': []}])

       restore_layer = []
       if marked_state[1] == '0':
           restore_layer.append({'name': 'x', 'qubits': [0], 'params': []})
       if marked_state[0] == '0':
           restore_layer.append({'name': 'x', 'qubits': [1], 'params': []})
       if restore_layer:
           layers.append(restore_layer)

       # Diffusion 算子
       layers.append([
           {'name': 'h', 'qubits': [0], 'params': []},
           {'name': 'h', 'qubits': [1], 'params': []},
       ])
       layers.append([
           {'name': 'x', 'qubits': [0], 'params': []},
           {'name': 'x', 'qubits': [1], 'params': []},
       ])
       layers.append([{'name': 'cz', 'qubits': [0, 1], 'params': []}])
       layers.append([
           {'name': 'x', 'qubits': [0], 'params': []},
           {'name': 'x', 'qubits': [1], 'params': []},
       ])
       layers.append([
           {'name': 'h', 'qubits': [0], 'params': []},
           {'name': 'h', 'qubits': [1], 'params': []},
       ])

       return Circuit.from_layers(layers, n_qubits=2)

   # 搜索 '11'
   qc = grover_search_2qubit('11')
   print("Grover 搜索电路:")
   print(qc.draw())

   sim = StatevectorSimulator()
   result = sim.run(qc, shots=1000)
   print(f"搜索结果: {result.counts}")
   # '11' 应该有最高概率


量子傅里叶变换 (QFT)
--------------------

QFT 是许多量子算法的核心组件：

.. code-block:: python

   from janus.circuit import Circuit
   import numpy as np

   def qft(n_qubits):
       """
       量子傅里叶变换

       Args:
           n_qubits: 量子比特数

       Returns:
           Circuit: QFT 电路
       """
       layers = []

       for i in range(n_qubits):
           # Hadamard 门
           layers.append([{'name': 'h', 'qubits': [i], 'params': []}])

           # 受控旋转门
           for j in range(i + 1, n_qubits):
               angle = np.pi / (2 ** (j - i))
               layers.append([{'name': 'cp', 'qubits': [j, i], 'params': [angle]}])

       # 交换量子比特顺序
       for i in range(n_qubits // 2):
           layers.append([{'name': 'swap', 'qubits': [i, n_qubits - 1 - i], 'params': []}])

       return Circuit.from_layers(layers, n_qubits=n_qubits)

   # 创建 4 量子比特 QFT
   qc = qft(4)
   print("QFT 电路:")
   print(qc.draw())
   print(f"电路深度: {qc.depth}")
   print(f"门数量: {qc.n_gates}")

变分量子本征求解器 (VQE) 简化版
-------------------------------

VQE 是一种混合量子-经典算法：

.. code-block:: python

   from janus.circuit import Circuit, Parameter
   from janus.simulator import StatevectorSimulator
   import numpy as np

   def create_ansatz(n_qubits, n_layers):
       """
       创建参数化 ansatz 电路
       """
       layers = []
       params = []

       for layer_idx in range(n_layers):
           # 旋转层
           rotation_layer = []
           for i in range(n_qubits):
               theta = Parameter(f'theta_{layer_idx}_{i}')
               params.append(theta)
               rotation_layer.append({'name': 'ry', 'qubits': [i], 'params': [theta]})
           layers.append(rotation_layer)

           # 纠缠层
           for i in range(n_qubits - 1):
               layers.append([{'name': 'cx', 'qubits': [i, i + 1], 'params': []}])

       qc = Circuit.from_layers(layers, n_qubits=n_qubits)
       return qc, params

   def compute_expectation(qc, params, param_values, observable):
       """
       计算期望值 <ψ|H|ψ>
       """
       sim = StatevectorSimulator()
       param_dict = {p.name: v for p, v in zip(params, param_values)}
       sv = sim.statevector(qc, parameter_binds=param_dict)
       return sv.expectation_value(observable).real

   # 创建 2 量子比特、2 层的 ansatz
   qc, params = create_ansatz(2, 2)
   print("VQE Ansatz 电路:")
   print(qc.draw())
   print(f"参数数量: {len(params)}")

   # 定义简单的哈密顿量 H = Z⊗Z
   Z = np.array([[1, 0], [0, -1]])
   ZZ = np.kron(Z, Z)

   # 计算一组参数的期望值
   param_values = [0.1, 0.2, 0.3, 0.4]
   energy = compute_expectation(qc, params, param_values, ZZ)
   print(f"能量期望值: {energy:.4f}")


噪声对量子电路的影响
--------------------

比较理想和噪声模拟的结果：

.. code-block:: python

   from janus.circuit import Circuit
   from janus.simulator import (
       StatevectorSimulator,
       NoisySimulator,
       NoiseModel,
       depolarizing_channel
   )

   # 创建 Bell 态电路
   qc = Circuit.from_layers([
       [{'name': 'h', 'qubits': [0], 'params': []}],
       [{'name': 'cx', 'qubits': [0, 1], 'params': []}],
   ], n_qubits=2)

   # 理想模拟
   ideal_sim = StatevectorSimulator(seed=42)
   ideal_result = ideal_sim.run(qc, shots=10000)

   # 噪声模拟（不同噪声级别）
   for error_rate in [0.001, 0.01, 0.05, 0.1]:
       noise_model = NoiseModel()
       noise_model.add_all_qubit_quantum_error(
           depolarizing_channel(error_rate),
           ['h', 'cx']
       )

       noisy_sim = NoisySimulator(noise_model, seed=42)
       noisy_result = noisy_sim.run(qc, shots=10000)

       # 计算保真度（正确结果的比例）
       correct = noisy_result.counts.get('00', 0) + noisy_result.counts.get('11', 0)
       fidelity = correct / 10000

       print(f"错误率 {error_rate*100:.1f}%: 保真度 = {fidelity:.3f}")

电路优化示例
------------

展示电路优化的效果：

.. code-block:: python

   from janus.circuit import Circuit
   from janus.compiler import compile_circuit
   from janus.simulator import StatevectorSimulator
   import numpy as np

   # 创建一个有冗余的电路
   qc = Circuit.from_layers([
       [{'name': 'h', 'qubits': [0], 'params': []}],
       [{'name': 'h', 'qubits': [0], 'params': []}],              # 冗余 H 门
       [{'name': 'rz', 'qubits': [1], 'params': [0.785]}],        # π/4
       [{'name': 'rz', 'qubits': [1], 'params': [0.785]}],        # 可合并
       [{'name': 'rz', 'qubits': [1], 'params': [0.785]}],        # 可合并
       [{'name': 'x', 'qubits': [2], 'params': []}],
       [{'name': 'x', 'qubits': [2], 'params': []}],              # 逆门对
       [{'name': 'cx', 'qubits': [0, 1], 'params': []}],
       [{'name': 'cx', 'qubits': [1, 2], 'params': []}],
   ], n_qubits=3)

   print("优化前:")
   print(qc.draw())
   print(f"门数量: {qc.n_gates}")

   # 优化
   optimized = compile_circuit(qc, optimization_level=2)

   print("\n优化后:")
   print(optimized.draw())
   print(f"门数量: {optimized.n_gates}")

   # 验证等价性
   sim = StatevectorSimulator()
   sv1 = sim.statevector(qc)
   sv2 = sim.statevector(optimized)
   print(f"\n电路等价: {sv1.equiv(sv2)}")

常用门名称参考
--------------

以下是 ``from_layers`` 中可用的门名称：

**单比特门**

.. code-block:: python

   # Pauli 门
   {'name': 'id', 'qubits': [0], 'params': []}   # I 门
   {'name': 'x', 'qubits': [0], 'params': []}    # X 门
   {'name': 'y', 'qubits': [0], 'params': []}    # Y 门
   {'name': 'z', 'qubits': [0], 'params': []}    # Z 门

   # Clifford 门
   {'name': 'h', 'qubits': [0], 'params': []}    # Hadamard
   {'name': 's', 'qubits': [0], 'params': []}    # S 门
   {'name': 'sdg', 'qubits': [0], 'params': []}  # S† 门
   {'name': 't', 'qubits': [0], 'params': []}    # T 门
   {'name': 'tdg', 'qubits': [0], 'params': []}  # T† 门
   {'name': 'sx', 'qubits': [0], 'params': []}   # √X 门

   # 旋转门
   {'name': 'rx', 'qubits': [0], 'params': [θ]}  # RX(θ)
   {'name': 'ry', 'qubits': [0], 'params': [θ]}  # RY(θ)
   {'name': 'rz', 'qubits': [0], 'params': [θ]}  # RZ(θ)
   {'name': 'p', 'qubits': [0], 'params': [λ]}   # Phase(λ)
   {'name': 'u', 'qubits': [0], 'params': [θ, φ, λ]}  # U(θ, φ, λ)

**两比特门**

.. code-block:: python

   {'name': 'cx', 'qubits': [0, 1], 'params': []}    # CNOT
   {'name': 'cy', 'qubits': [0, 1], 'params': []}    # CY
   {'name': 'cz', 'qubits': [0, 1], 'params': []}    # CZ
   {'name': 'ch', 'qubits': [0, 1], 'params': []}    # CH
   {'name': 'swap', 'qubits': [0, 1], 'params': []}  # SWAP
   {'name': 'iswap', 'qubits': [0, 1], 'params': []} # iSWAP

   # 受控旋转门
   {'name': 'crx', 'qubits': [0, 1], 'params': [θ]}  # CRX(θ)
   {'name': 'cry', 'qubits': [0, 1], 'params': [θ]}  # CRY(θ)
   {'name': 'crz', 'qubits': [0, 1], 'params': [θ]}  # CRZ(θ)
   {'name': 'cp', 'qubits': [0, 1], 'params': [θ]}   # CPhase(θ)

   # 两比特旋转门
   {'name': 'rxx', 'qubits': [0, 1], 'params': [θ]}  # RXX(θ)
   {'name': 'ryy', 'qubits': [0, 1], 'params': [θ]}  # RYY(θ)
   {'name': 'rzz', 'qubits': [0, 1], 'params': [θ]}  # RZZ(θ)

**三比特及多比特门**

.. code-block:: python

   {'name': 'ccx', 'qubits': [0, 1, 2], 'params': []}   # Toffoli
   {'name': 'ccz', 'qubits': [0, 1, 2], 'params': []}   # CCZ
   {'name': 'cswap', 'qubits': [0, 1, 2], 'params': []} # Fredkin
