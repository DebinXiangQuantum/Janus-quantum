janus.circuit
=============

量子电路模块，提供电路构建、操作和表示。

核心类
------

Circuit
~~~~~~~

.. autoclass:: janus.circuit.Circuit
   :members:
   :undoc-members:
   :show-inheritance:

SeperatableCircuit
~~~~~~~~~~~~~~~~~~

.. autoclass:: janus.circuit.SeperatableCircuit
   :members:
   :undoc-members:
   :show-inheritance:

Gate
~~~~

.. autoclass:: janus.circuit.Gate
   :members:
   :undoc-members:
   :show-inheritance:

ControlledGate
~~~~~~~~~~~~~~

.. autoclass:: janus.circuit.ControlledGate
   :members:
   :undoc-members:
   :show-inheritance:

Instruction
~~~~~~~~~~~

.. autoclass:: janus.circuit.Instruction
   :members:
   :undoc-members:
   :show-inheritance:

Layer
~~~~~

.. autoclass:: janus.circuit.Layer
   :members:
   :undoc-members:
   :show-inheritance:

量子比特和经典比特
------------------

Qubit
~~~~~

.. autoclass:: janus.circuit.Qubit
   :members:
   :undoc-members:

QuantumRegister
~~~~~~~~~~~~~~~

.. autoclass:: janus.circuit.QuantumRegister
   :members:
   :undoc-members:

Clbit
~~~~~

.. autoclass:: janus.circuit.Clbit
   :members:
   :undoc-members:

ClassicalRegister
~~~~~~~~~~~~~~~~~

.. autoclass:: janus.circuit.ClassicalRegister
   :members:
   :undoc-members:

参数化
------

Parameter
~~~~~~~~~

.. autoclass:: janus.circuit.Parameter
   :members:
   :undoc-members:

ParameterExpression
~~~~~~~~~~~~~~~~~~~

.. autoclass:: janus.circuit.ParameterExpression
   :members:
   :undoc-members:

DAG
---

DAGCircuit
~~~~~~~~~~

.. autoclass:: janus.circuit.DAGCircuit
   :members:
   :undoc-members:

DAGDependency
~~~~~~~~~~~~~

.. autoclass:: janus.circuit.DAGDependency
   :members:
   :undoc-members:

转换函数
~~~~~~~~

.. autofunction:: janus.circuit.circuit_to_dag

.. autofunction:: janus.circuit.dag_to_circuit

.. autofunction:: janus.circuit.circuit_to_dag_dependency

文件 I/O
--------

.. autofunction:: janus.circuit.load_circuit

.. autofunction:: janus.circuit.list_circuits

.. autofunction:: janus.circuit.get_circuits_dir

标准门库
--------

单比特门
~~~~~~~~

.. autoclass:: janus.circuit.HGate
   :members:
   :undoc-members:

.. autoclass:: janus.circuit.XGate
   :members:
   :undoc-members:

.. autoclass:: janus.circuit.YGate
   :members:
   :undoc-members:

.. autoclass:: janus.circuit.ZGate
   :members:
   :undoc-members:

.. autoclass:: janus.circuit.RXGate
   :members:
   :undoc-members:

.. autoclass:: janus.circuit.RYGate
   :members:
   :undoc-members:

.. autoclass:: janus.circuit.RZGate
   :members:
   :undoc-members:

两比特门
~~~~~~~~

.. autoclass:: janus.circuit.CXGate
   :members:
   :undoc-members:

.. autoclass:: janus.circuit.CZGate
   :members:
   :undoc-members:

.. autoclass:: janus.circuit.SwapGate
   :members:
   :undoc-members:

多比特门
~~~~~~~~

.. autoclass:: janus.circuit.CCXGate
   :members:
   :undoc-members:

.. autoclass:: janus.circuit.MCXGate
   :members:
   :undoc-members:
