janus.encode
============

量子态编码模块。

编码函数
--------

schmidt_encode
~~~~~~~~~~~~~~

.. autofunction:: janus.encode.schmidt_encode

bidrc_encode
~~~~~~~~~~~~

.. autofunction:: janus.encode.bidrc_encode

efficient_sparse
~~~~~~~~~~~~~~~~

.. autofunction:: janus.encode.efficient_sparse

结果类
------

SchmidtEncodeResult
~~~~~~~~~~~~~~~~~~~

.. autoclass:: janus.encode.SchmidtEncodeResult
   :members:
   :undoc-members:

EfficientSparseResult
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: janus.encode.EfficientSparseResult
   :members:
   :undoc-members:

工具函数
--------

.. autofunction:: janus.encode._schmidt

.. autofunction:: janus.encode._build_state_dict

.. autofunction:: janus.encode._complete_to_unitary

.. autofunction:: janus.encode._apply_unitary

使用示例
--------

Schmidt 编码
~~~~~~~~~~~~

.. code-block:: python

   from janus.encode import schmidt_encode
   import numpy as np

   data = [1/np.sqrt(2), 1/np.sqrt(2), 0, 0]
   circuit = schmidt_encode(q_size=4, data=data, cutoff=1e-4)

高效稀疏编码
~~~~~~~~~~~~

.. code-block:: python

   from janus.encode import efficient_sparse

   result = efficient_sparse(data, n_qubits=4)
