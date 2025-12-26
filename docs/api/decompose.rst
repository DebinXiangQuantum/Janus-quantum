janus.decompose
===============

量子门分解模块。

分解函数
--------

decompose_one_qubit
~~~~~~~~~~~~~~~~~~~

.. autofunction:: janus.decompose.decompose_one_qubit

decompose_two_qubit_gate
~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: janus.decompose.decompose_two_qubit_gate

decompose_multi_control_toffoli
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: janus.decompose.decompose_multi_control_toffoli

decompose_controlled_gate
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: janus.decompose.decompose_controlled_gate

decompose_kak
~~~~~~~~~~~~~

.. autofunction:: janus.decompose.decompose_kak

convert_circuit_to_instruction_set
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: janus.decompose.convert_circuit_to_instruction_set

分解器类
--------

OneQubitEulerDecomposer
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: janus.decompose.OneQubitEulerDecomposer
   :members:
   :undoc-members:
   :show-inheritance:

异常
----

.. autoexception:: janus.decompose.DecomposeError

.. autoexception:: janus.decompose.UnsupportedMethodError

.. autoexception:: janus.decompose.GateNotSupportedError

.. autoexception:: janus.decompose.ParameterError

.. autoexception:: janus.decompose.CircuitError

使用示例
--------

单比特门分解
~~~~~~~~~~~~

.. code-block:: python

   from janus.decompose import decompose_one_qubit
   import numpy as np

   # 分解任意单比特酉矩阵
   U = np.array([[1, 0], [0, 1j]])
   gates = decompose_one_qubit(U)

两比特门分解
~~~~~~~~~~~~

.. code-block:: python

   from janus.decompose import decompose_two_qubit_gate

   gates = decompose_two_qubit_gate(U)

KAK 分解
~~~~~~~~

.. code-block:: python

   from janus.decompose import decompose_kak

   result = decompose_kak(U)
