贡献指南
========

感谢你对 Janus 项目的关注！

开发环境
--------

.. code-block:: bash

   git clone https://github.com/your-org/janus.git
   cd janus
   pip install -e ".[dev]"

代码风格
--------

- 遵循 PEP 8
- 使用 Black 格式化
- 使用 isort 排序导入

.. code-block:: bash

   black janus/
   isort janus/

测试
----

.. code-block:: bash

   pytest test/

文档
----

.. code-block:: bash

   cd docs
   make html

提交 PR
-------

1. Fork 项目
2. 创建功能分支
3. 提交更改
4. 创建 Pull Request

问题反馈
--------

请在 GitHub Issues 中报告问题。
