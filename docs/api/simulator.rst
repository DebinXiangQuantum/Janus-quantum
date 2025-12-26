janus.simulator
===============

量子电路模拟器模块。

模拟器
------

StatevectorSimulator
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: janus.simulator.StatevectorSimulator
   :members:
   :undoc-members:
   :show-inheritance:

NoisySimulator
~~~~~~~~~~~~~~

.. autoclass:: janus.simulator.NoisySimulator
   :members:
   :undoc-members:
   :show-inheritance:

量子态
------

Statevector
~~~~~~~~~~~

.. autoclass:: janus.simulator.Statevector
   :members:
   :undoc-members:
   :show-inheritance:

DensityMatrix
~~~~~~~~~~~~~

.. autoclass:: janus.simulator.DensityMatrix
   :members:
   :undoc-members:
   :show-inheritance:

结果
----

SimulatorResult
~~~~~~~~~~~~~~~

.. autoclass:: janus.simulator.SimulatorResult
   :members:
   :undoc-members:

Counts
~~~~~~

.. autoclass:: janus.simulator.Counts
   :members:
   :undoc-members:

噪声
----

NoiseModel
~~~~~~~~~~

.. autoclass:: janus.simulator.NoiseModel
   :members:
   :undoc-members:

NoiseChannel
~~~~~~~~~~~~

.. autoclass:: janus.simulator.NoiseChannel
   :members:
   :undoc-members:

噪声信道函数
~~~~~~~~~~~~

.. autofunction:: janus.simulator.depolarizing_channel

.. autofunction:: janus.simulator.amplitude_damping_channel

.. autofunction:: janus.simulator.phase_damping_channel

.. autofunction:: janus.simulator.bit_flip_channel

.. autofunction:: janus.simulator.phase_flip_channel

.. autofunction:: janus.simulator.thermal_relaxation_channel

.. autofunction:: janus.simulator.readout_error_channel

异常
----

.. autoexception:: janus.simulator.SimulatorError

.. autoexception:: janus.simulator.InvalidStateError

.. autoexception:: janus.simulator.InvalidCircuitError

.. autoexception:: janus.simulator.ParameterBindingError
