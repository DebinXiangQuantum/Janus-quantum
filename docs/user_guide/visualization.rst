电路可视化
==========

Janus 支持文本和图像两种电路可视化方式。

文本绘图
--------

基本用法
~~~~~~~~

.. code-block:: python

   from janus.circuit import Circuit

   qc = Circuit(2)
   qc.h(0)
   qc.cx(0, 1)

   print(qc.draw())

输出::

   q0: ─H─●─
         │
   q1: ───X─

折叠选项
~~~~~~~~

.. code-block:: python

   # 每行最多 3 层
   print(qc.draw(fold=3))

   # 指定行宽
   print(qc.draw(line_length=80))

   # 禁用折叠
   print(qc.draw(fold=-1))

图像导出
--------

PNG 导出
~~~~~~~~

.. code-block:: python

   qc.draw(output='png', filename='circuit.png')

   # 自定义尺寸和分辨率
   qc.draw(output='png', filename='circuit.png', figsize=(12, 6), dpi=200)

Matplotlib 图形
~~~~~~~~~~~~~~~

.. code-block:: python

   fig = qc.draw(output='mpl')
   fig.savefig('circuit.pdf')

命令行工具
----------

.. code-block:: bash

   # 查看电路信息
   python -m janus.circuit.cli info circuit.json
   python -m janus.circuit.cli info circuit.json -v  # 详细信息

   # 绘制电路
   python -m janus.circuit.cli draw circuit.json
   python -m janus.circuit.cli draw circuit.json -o output.png

   # 测试电路
   python -m janus.circuit.cli test circuit.json
