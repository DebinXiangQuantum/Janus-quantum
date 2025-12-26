janus.compiler
==============

量子电路编译器模块。

编译函数
--------

.. autofunction:: janus.compiler.compile_circuit

优化 Pass
---------

BasePass
~~~~~~~~

.. autoclass:: janus.compiler.BasePass
   :members:
   :undoc-members:
   :show-inheritance:

RemoveIdentityPass
~~~~~~~~~~~~~~~~~~

.. autoclass:: janus.compiler.RemoveIdentityPass
   :members:
   :undoc-members:
   :show-inheritance:

MergeRotationsPass
~~~~~~~~~~~~~~~~~~

.. autoclass:: janus.compiler.MergeRotationsPass
   :members:
   :undoc-members:
   :show-inheritance:

CancelInversesPass
~~~~~~~~~~~~~~~~~~

.. autoclass:: janus.compiler.CancelInversesPass
   :members:
   :undoc-members:
   :show-inheritance:

使用示例
--------

基本编译
~~~~~~~~

.. code-block:: python

   from janus.compiler import compile_circuit

   optimized = compile_circuit(qc, optimization_level=2)

自定义 Pass
~~~~~~~~~~~

.. code-block:: python

   from janus.compiler.passes import (
       RemoveIdentityPass,
       CancelInversesPass,
       MergeRotationsPass
   )

   optimized = compile_circuit(qc, passes=[
       RemoveIdentityPass(),
       CancelInversesPass(),
       MergeRotationsPass(),
   ])
