安装指南
========

系统要求
--------

- Python 3.9 或更高版本
- pip 包管理器

从 PyPI 安装
------------

推荐使用 pip 安装：

.. code-block:: bash

   pip install janus-quantum

从源码安装
----------

如果需要最新开发版本：

.. code-block:: bash

   git clone https://github.com/DebinXiangQuantum/Janus-compiler.git
   cd janus
   pip install -e .

依赖项
------

核心依赖（自动安装）：

- ``numpy>=1.16.0`` - 数值计算
- ``scipy>=1.5.0`` - 科学计算
- ``rustworkx>=0.12.0`` - 图算法

可选依赖：

.. code-block:: bash

   # 可视化支持
   pip install matplotlib>=3.0.0

   # 完整安装（包含所有可选依赖）
   pip install janus-quantum[full]

验证安装
--------

.. code-block:: python

   import janus
   from janus.circuit import Circuit

   qc = Circuit(2)
   qc.h(0)
   qc.cx(0, 1)
   print(qc.draw())

如果看到电路图输出，说明安装成功。

常见问题
--------

**Q: 安装时提示 rustworkx 编译失败？**

A: 确保系统已安装 Rust 编译器，或尝试：

.. code-block:: bash

   pip install --upgrade pip
   pip install janus-quantum

**Q: 如何在虚拟环境中安装？**

A: 推荐使用 venv 或 conda：

.. code-block:: bash

   python -m venv janus-env
   source janus-env/bin/activate  # Linux/Mac
   # 或 janus-env\Scripts\activate  # Windows
   pip install janus-quantum
