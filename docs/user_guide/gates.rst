量子门库
========

Janus 提供 60+ 种标准量子门。

单比特门
--------

Pauli 门
~~~~~~~~

.. list-table::
   :header-rows: 1

   * - 门
     - 方法
     - 说明
   * - I
     - ``qc.id(q)``
     - 恒等门
   * - X
     - ``qc.x(q)``
     - Pauli-X (NOT)
   * - Y
     - ``qc.y(q)``
     - Pauli-Y
   * - Z
     - ``qc.z(q)``
     - Pauli-Z

Clifford 门
~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - 门
     - 方法
     - 说明
   * - H
     - ``qc.h(q)``
     - Hadamard
   * - S
     - ``qc.s(q)``
     - √Z
   * - S†
     - ``qc.sdg(q)``
     - S 共轭转置
   * - √X
     - ``qc.sx(q)``
     - √X

旋转门
~~~~~~

.. list-table::
   :header-rows: 1

   * - 门
     - 方法
     - 参数
   * - RX
     - ``qc.rx(θ, q)``
     - θ: 旋转角度
   * - RY
     - ``qc.ry(θ, q)``
     - θ: 旋转角度
   * - RZ
     - ``qc.rz(θ, q)``
     - θ: 旋转角度
   * - P
     - ``qc.p(λ, q)``
     - λ: 相位
   * - U
     - ``qc.u(θ, φ, λ, q)``
     - 通用单比特门

两比特门
--------

基本两比特门
~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - 门
     - 方法
     - 说明
   * - CX
     - ``qc.cx(c, t)``
     - CNOT
   * - CY
     - ``qc.cy(c, t)``
     - 受控 Y
   * - CZ
     - ``qc.cz(c, t)``
     - 受控 Z
   * - CH
     - ``qc.ch(c, t)``
     - 受控 H
   * - SWAP
     - ``qc.swap(q1, q2)``
     - 交换门
   * - iSWAP
     - ``qc.iswap(q1, q2)``
     - iSWAP

受控旋转门
~~~~~~~~~~

.. code-block:: python

   qc.crx(θ, c, t)  # 受控 RX
   qc.cry(θ, c, t)  # 受控 RY
   qc.crz(θ, c, t)  # 受控 RZ
   qc.cp(θ, c, t)   # 受控 Phase

两比特旋转门
~~~~~~~~~~~~

.. code-block:: python

   qc.rxx(θ, q1, q2)  # XX 旋转
   qc.ryy(θ, q1, q2)  # YY 旋转
   qc.rzz(θ, q1, q2)  # ZZ 旋转
   qc.rzx(θ, q1, q2)  # ZX 旋转

多比特门
--------

三比特门
~~~~~~~~

.. code-block:: python

   qc.ccx(c1, c2, t)     # Toffoli
   qc.ccz(c1, c2, t)     # 双控制 Z
   qc.cswap(c, t1, t2)   # Fredkin

多控制门
~~~~~~~~

.. code-block:: python

   qc.mcx([0, 1], 2)              # 多控制 X
   qc.mcp(np.pi/4, [0, 1], 2)     # 多控制 Phase
   qc.mcrx(np.pi/4, [0, 1], 2)    # 多控制 RX
   qc.mcry(np.pi/3, [0, 1, 2], 3) # 多控制 RY
   qc.mcrz(np.pi/2, [0], 1)       # 多控制 RZ

链式创建受控门
~~~~~~~~~~~~~~

.. code-block:: python

   from janus.circuit.library import U3Gate, RXGate, HGate

   qc.gate(RXGate(np.pi/4), 2).control(0)
   qc.gate(HGate(), 2).control([0, 1])
   qc.gate(U3Gate(np.pi/4, 0, 0), 3).control([0, 1, 2])

特殊操作
--------

.. list-table::
   :header-rows: 1

   * - 操作
     - 方法
     - 说明
   * - Barrier
     - ``qc.barrier()``
     - 屏障
   * - Measure
     - ``qc.measure(q, c)``
     - 测量
   * - Reset
     - ``qc.reset(q)``
     - 重置
   * - Delay
     - ``qc.delay(duration, q)``
     - 延迟

参数化门
--------

.. code-block:: python

   from janus.circuit import Parameter

   theta = Parameter('theta')
   phi = Parameter('phi')

   qc = Circuit(2)
   qc.rx(theta, 0)
   qc.ry(phi, 1)

   # 检查参数
   print(qc.parameters)
   print(qc.is_parameterized())

   # 绑定参数
   bound_qc = qc.bind_parameters({theta: np.pi/2, phi: np.pi/4})
