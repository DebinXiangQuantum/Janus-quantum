电路优化
========

Janus 提供多级电路优化功能。

基本编译
--------

.. code-block:: python

   from janus.compiler import compile_circuit

   qc = Circuit(2)
   qc.h(0)
   qc.h(0)  # 冗余
   qc.rz(np.pi/4, 0)
   qc.rz(np.pi/4, 0)  # 会合并

   optimized = compile_circuit(qc, optimization_level=2)

优化级别
--------

.. list-table::
   :header-rows: 1

   * - 级别
     - 内容
   * - 0
     - 无优化
   * - 1
     - 移除恒等门、消除逆门对
   * - 2
     - 级别1 + 合并连续旋转门

自定义 Pass
-----------

.. code-block:: python

   from janus.compiler.passes import (
       CancelInversesPass,
       MergeRotationsPass,
       RemoveIdentityPass
   )

   optimized = compile_circuit(qc, passes=[
       RemoveIdentityPass(),
       CancelInversesPass(),
       MergeRotationsPass(),
   ])

高级优化
--------

Janus 的 ``optimize`` 模块提供更多高级优化技术。

Clifford+Rz 优化
~~~~~~~~~~~~~~~~

.. code-block:: python

   from janus.optimize import TChinMerger, CliffordMerger

门融合
~~~~~~

.. code-block:: python

   from janus.optimize import (
       ConsolidateBlocks,
       Optimize1qGates,
       Collect2qBlocks
   )

交换性优化
~~~~~~~~~~

.. code-block:: python

   from janus.optimize import (
       CommutativeCancellation,
       InverseCancellation
   )

模板匹配
~~~~~~~~

.. code-block:: python

   from janus.optimize import (
       TemplateOptimization,
       TemplateMatching
   )

电路分析
--------

.. code-block:: python

   from janus.optimize import Depth, CountOps, Size

   # 分析电路深度
   depth_analyzer = Depth()

   # 统计门数量
   count_analyzer = CountOps()
