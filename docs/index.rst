Janus 量子电路框架
==================

.. image:: https://img.shields.io/badge/python-3.9+-blue.svg
   :target: https://www.python.org/downloads/

.. image:: https://img.shields.io/badge/license-MIT-green.svg
   :target: https://opensource.org/licenses/MIT

**Janus** 是一个轻量级、高性能的量子电路构建、模拟和编译框架，专为量子计算研究和教学设计。

为什么选择 Janus？
------------------

🚀 **轻量高效**
   纯 Python 实现，依赖少，启动快。无需复杂的安装过程，几秒钟即可开始使用。

🔧 **60+ 量子门**
   完整的标准门库，从基础的 Pauli 门到复杂的多控制门，支持参数化电路。

📊 **多种模拟器**
   状态向量模拟、密度矩阵模拟、噪声模拟，满足不同场景需求。

⚡ **电路优化**
   多级优化 Pass，自动门合并与消除，提升电路执行效率。

🎨 **可视化**
   文本和图像两种电路绘制方式，方便调试和展示。

📁 **JSON 格式**
   简洁的电路存储和交换格式，易于与其他工具集成。

5 分钟快速体验
--------------

**安装**

.. code-block:: bash

   pip install janus-quantum

**创建你的第一个量子电路**

.. code-block:: python

   from janus.circuit import Circuit
   from janus.simulator import StatevectorSimulator

   # 创建 2 量子比特的 Bell 态电路
   qc = Circuit.from_layers([
       [{'name': 'h', 'qubits': [0], 'params': []}],      # H 门
       [{'name': 'cx', 'qubits': [0, 1], 'params': []}],  # CNOT 门
   ], n_qubits=2)

   # 查看电路结构
   print(qc.draw())
   # 输出:
   # q0: ─H─●─
   #        │
   # q1: ───X─

   # 模拟运行
   sim = StatevectorSimulator()
   result = sim.run(qc, shots=1000)
   print(result.counts)  # {'00': ~500, '11': ~500}

**恭喜！** 你刚刚创建并模拟了一个量子纠缠态。

典型应用场景
------------

**量子算法研究**
   快速原型设计和验证量子算法，如 Grover 搜索、VQE、QAOA 等。

**量子计算教学**
   简洁的 API 设计，适合教学演示和学生实验。

**电路优化研究**
   丰富的优化 Pass 和分析工具，支持电路优化算法研究。

**噪声模拟**
   真实噪声模型，帮助理解量子计算中的错误和退相干。

文档目录
--------

.. toctree::
   :maxdepth: 2
   :caption: 🚀 入门指南

   getting_started/installation
   getting_started/quickstart
   getting_started/concepts

.. toctree::
   :maxdepth: 2
   :caption: 📖 用户指南

   user_guide/circuits
   user_guide/gates
   user_guide/simulation
   user_guide/optimization
   user_guide/visualization
   user_guide/examples

.. toctree::
   :maxdepth: 2
   :caption: 📚 API 参考

   api/circuit
   api/simulator
   api/compiler
   api/decompose
   api/optimize
   api/encode

.. toctree::
   :maxdepth: 1
   :caption: 📋 其他

   changelog
   contributing

索引
----

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
