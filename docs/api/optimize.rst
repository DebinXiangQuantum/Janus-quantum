janus.optimize
==============

高级电路优化模块。

基类
----

TransformationPass
~~~~~~~~~~~~~~~~~~

.. autoclass:: janus.optimize.TransformationPass
   :members:
   :undoc-members:
   :show-inheritance:

AnalysisPass
~~~~~~~~~~~~

.. autoclass:: janus.optimize.AnalysisPass
   :members:
   :undoc-members:
   :show-inheritance:

Clifford+Rz 优化
----------------

.. autoclass:: janus.optimize.OptimizeCliffordT
   :members:
   :undoc-members:

.. autoclass:: janus.optimize.TChinMerger
   :members:
   :undoc-members:

.. autoclass:: janus.optimize.CliffordMerger
   :members:
   :undoc-members:

门融合优化
----------

.. autoclass:: janus.optimize.ConsolidateBlocks
   :members:
   :undoc-members:

.. autoclass:: janus.optimize.Optimize1qGates
   :members:
   :undoc-members:

.. autoclass:: janus.optimize.Collect2qBlocks
   :members:
   :undoc-members:

交换性优化
----------

.. autoclass:: janus.optimize.CommutativeCancellation
   :members:
   :undoc-members:

.. autoclass:: janus.optimize.InverseCancellation
   :members:
   :undoc-members:

模板匹配
--------

.. autoclass:: janus.optimize.TemplateOptimization
   :members:
   :undoc-members:

.. autoclass:: janus.optimize.TemplateMatching
   :members:
   :undoc-members:

合成算法
--------

KAK 分解
~~~~~~~~

.. autoclass:: janus.optimize.TwoQubitWeylDecomposition
   :members:
   :undoc-members:

.. autoclass:: janus.optimize.TwoQubitBasisDecomposer
   :members:
   :undoc-members:

Clifford 合成
~~~~~~~~~~~~~

.. autofunction:: janus.optimize.synthesize_clifford_circuit

.. autofunction:: janus.optimize.synthesize_clifford_bravyi_maslov

CNOT 优化
~~~~~~~~~

.. autofunction:: janus.optimize.synthesize_cnot_count_pmh

.. autofunction:: janus.optimize.synthesize_cnot_depth_lnn_kms

分析 Pass
---------

.. autoclass:: janus.optimize.Depth
   :members:
   :undoc-members:

.. autoclass:: janus.optimize.Width
   :members:
   :undoc-members:

.. autoclass:: janus.optimize.Size
   :members:
   :undoc-members:

.. autoclass:: janus.optimize.CountOps
   :members:
   :undoc-members:

.. autoclass:: janus.optimize.ResourceEstimation
   :members:
   :undoc-members:
