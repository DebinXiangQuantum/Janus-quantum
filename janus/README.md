# Janus 量子电路框架

轻量级量子电路构建、表示和编译框架，兼容 Qiskit 标准门库。

## 安装

```bash
pip install numpy matplotlib  # matplotlib 用于电路可视化

```

## 快速开始

### 创建电路

```python
from janus.circuit import Circuit
import numpy as np

# 创建 2 量子比特、1 经典比特电路
qc = Circuit(2, 1, name="Bell")

# 添加门
qc.h(0)           # Hadamard 门
qc.cx(0, 1)       # CNOT 门
qc.rx(np.pi/2, 0) # RX 旋转门
qc.measure(0, 0)  # 测量

print(qc)
print(qc.draw())
```

## 支持的量子门 (60+)

### 单比特 Pauli 门
| 门 | 类名 | 快捷方法 | 说明 |
|---|------|---------|------|
| I | `IGate` | `qc.id(q)` | 恒等门 |
| X | `XGate` | `qc.x(q)` | Pauli-X (NOT) |
| Y | `YGate` | `qc.y(q)` | Pauli-Y |
| Z | `ZGate` | `qc.z(q)` | Pauli-Z |

### Clifford 门
| 门 | 类名 | 快捷方法 | 说明 |
|---|------|---------|------|
| H | `HGate` | `qc.h(q)` | Hadamard |
| S | `SGate` | `qc.s(q)` | √Z |
| S† | `SdgGate` | `qc.sdg(q)` | S 的共轭转置 |
| T | `TGate` | `qc.t(q)` | √S |
| T† | `TdgGate` | `qc.tdg(q)` | T 的共轭转置 |
| √X | `SXGate` | `qc.sx(q)` | √X |
| √X† | `SXdgGate` | `qc.sxdg(q)` | √X 的共轭转置 |

### 单比特旋转门
| 门 | 类名 | 快捷方法 | 参数 |
|---|------|---------|------|
| RX | `RXGate` | `qc.rx(θ, q)` | θ: 旋转角度 |
| RY | `RYGate` | `qc.ry(θ, q)` | θ: 旋转角度 |
| RZ | `RZGate` | `qc.rz(θ, q)` | θ: 旋转角度 |
| P | `PhaseGate` | `qc.p(λ, q)` | λ: 相位角度 |
| U1 | `U1Gate` | `qc.u1(λ, q)` | λ: 相位 |
| U2 | `U2Gate` | `qc.u2(φ, λ, q)` | φ, λ: 角度 |
| U3 | `U3Gate` | `qc.u3(θ, φ, λ, q)` | θ, φ, λ: 角度 |
| U | `UGate` | `qc.u(θ, φ, λ, q)` | 通用单比特门 |
| R | `RGate` | - | θ, φ: 任意轴旋转 |

### 两比特旋转门
| 门 | 类名 | 说明 |
|---|------|------|
| RXX | `RXXGate` | XX 旋转 |
| RYY | `RYYGate` | YY 旋转 |
| RZZ | `RZZGate` | ZZ 旋转 |
| RZX | `RZXGate` | ZX 旋转 |

### 两比特门
| 门 | 类名 | 快捷方法 | 说明 |
|---|------|---------|------|
| CX | `CXGate` | `qc.cx(c, t)` | CNOT |
| CY | `CYGate` | `qc.cy(c, t)` | 受控 Y |
| CZ | `CZGate` | `qc.cz(c, t)` | 受控 Z |
| CH | `CHGate` | `qc.ch(c, t)` | 受控 H |
| CS | `CSGate` | - | 受控 S |
| CS† | `CSdgGate` | - | 受控 S† |
| CSX | `CSXGate` | - | 受控 √X |
| DCX | `DCXGate` | - | Double CX |
| ECR | `ECRGate` | - | Echoed Cross-Resonance |
| SWAP | `SwapGate` | `qc.swap(q1, q2)` | 交换门 |
| iSWAP | `iSwapGate` | `qc.iswap(q1, q2)` | iSWAP |

### 受控旋转门
| 门 | 类名 | 快捷方法 | 说明 |
|---|------|---------|------|
| CRX | `CRXGate` | `qc.crx(θ, c, t)` | 受控 RX |
| CRY | `CRYGate` | `qc.cry(θ, c, t)` | 受控 RY |
| CRZ | `CRZGate` | `qc.crz(θ, c, t)` | 受控 RZ |
| CP | `CPhaseGate` | `qc.cp(θ, c, t)` | 受控 Phase |
| CU1 | `CU1Gate` | - | 受控 U1 |
| CU3 | `CU3Gate` | - | 受控 U3 |
| CU | `CUGate` | - | 受控 U (带全局相位) |

### 三比特及多比特门
| 门 | 类名 | 快捷方法 | 说明 |
|---|------|---------|------|
| CCX | `CCXGate` | `qc.ccx(c1, c2, t)` | Toffoli |
| CCZ | `CCZGate` | `qc.ccz(c1, c2, t)` | 双控制 Z |
| CSWAP | `CSwapGate` | `qc.cswap(c, t1, t2)` | Fredkin |
| RCCX | `RCCXGate` | - | 简化 Toffoli |
| RC3X | `RC3XGate` | - | 简化三控制 X |
| C3X | `C3XGate` | - | 三控制 X |
| C4X | `C4XGate` | - | 四控制 X |

### 多控制门 (新增)
| 门 | 类名 | 说明 |
|---|------|------|
| C3SX | `C3SXGate` | 三控制 √X |
| MCX | `MCXGate` | 多控制 X (通用) |
| MCXGrayCode | `MCXGrayCode` | Gray code 实现 |
| MCXRecursive | `MCXRecursive` | 递归实现 |
| MCXVChain | `MCXVChain` | V-chain 实现 |
| MCPhase | `MCPhaseGate` | 多控制 Phase |
| MCU1 | `MCU1Gate` | 多控制 U1 |

### 特殊门
| 门 | 类名 | 说明 |
|---|------|------|
| XX-YY | `XXMinusYYGate` | XX-YY 相互作用 |
| XX+YY | `XXPlusYYGate` | XX+YY 相互作用 |
| GlobalPhase | `GlobalPhaseGate` | 全局相位 |

### 特殊操作
| 操作 | 类名 | 快捷方法 | 说明 |
|------|------|---------|------|
| Barrier | `Barrier` | `qc.barrier()` | 屏障 |
| Measure | `Measure` | `qc.measure(q, c)` | 测量 |
| Reset | `Reset` | `qc.reset(q)` | 重置 |
| Delay | `Delay` | - | 延迟 |

## 电路可视化

### 文本绘图

```python
print(qc.draw())
```

输出示例：
```
q0: ──H────●────RX(1.57)──
           │
q1: ───────X──────────────
```

### PNG 图像导出

```python
# 保存为 PNG 文件
qc.draw(output='png', filename='circuit.png')

# 自定义大小和分辨率
qc.draw(output='png', filename='circuit.png', figsize=(12, 6), dpi=200)

# 获取 matplotlib Figure 对象
fig = qc.draw(output='mpl')
fig.savefig('circuit.pdf')  # 保存为 PDF
fig.savefig('circuit.svg')  # 保存为 SVG
```

### 支持的门绘制符号
- 单比特门：方框 + 门名称
- 控制门：● (控制点) + ⊕ (目标)
- SWAP：× 符号
- 测量：M 符号
- 多控制门：多个 ● 连接

## DAG (有向无环图) 表示

### 基本 DAG 操作

```python
from janus.circuit.dag import circuit_to_dag, dag_to_circuit

# 电路转 DAG
dag = circuit_to_dag(qc)

# DAG 属性
print(dag.depth())       # 电路深度
print(dag.count_ops())   # 门统计 {'h': 1, 'cx': 1, ...}
print(dag.layers())      # 分层表示

# 遍历操作节点
for node in dag.op_nodes():
    print(node.name, node.qubits)

# 拓扑排序遍历
for node in dag.topological_op_nodes():
    print(node)

# DAG 转回电路
qc2 = dag_to_circuit(dag)
```

### DAG 高级功能

```python
# 祖先和后代查询
for node in dag.op_nodes():
    ancestors = dag.ancestors(node)      # 所有祖先节点
    descendants = dag.descendants(node)  # 所有后代节点
    break

# DAG 复制
dag_copy = dag.copy()

# 最长路径
path = dag.longest_path()

# 特定类型的操作
two_qubit_ops = list(dag.two_qubit_ops())
multi_qubit_ops = list(dag.multi_qubit_ops())
gate_nodes = list(dag.gate_nodes())  # 排除 measure/reset/barrier
```

### DAGDependency (交换性分析)

```python
from janus.circuit.dag import circuit_to_dag_dependency, dag_dependency_to_circuit

# 创建基于依赖关系的 DAG
dag_dep = circuit_to_dag_dependency(qc)

print(dag_dep.size())   # 节点数
print(dag_dep.depth())  # 深度

# 查询依赖关系
for node in dag_dep.get_nodes():
    succs = dag_dep.direct_successors(node.node_id)
    preds = dag_dep.direct_predecessors(node.node_id)

# 转回电路
qc2 = dag_dependency_to_circuit(dag_dep)
```

### 块操作

```python
from janus.circuit.dag import (
    BlockCollector, BlockSplitter, BlockCollapser, 
    split_block_into_layers
)

# 收集满足条件的块
dag = circuit_to_dag(qc)
collector = BlockCollector(dag)

# 收集所有单比特门块
single_qubit_blocks = collector.collect_all_matching_blocks(
    filter_fn=lambda n: len(n.qubits) == 1,
    min_block_size=2
)

# 收集所有两比特门块
two_qubit_blocks = collector.collect_all_matching_blocks(
    filter_fn=lambda n: len(n.qubits) == 2,
    split_blocks=True,      # 分割为不相交子块
    split_layers=False,     # 不分层
    collect_from_back=False # 从前向后收集
)

# 将块分割为层
for block in single_qubit_blocks:
    layers = split_block_into_layers(block)
```

## 参数化电路

```python
from janus.circuit import Circuit, Parameter
import numpy as np

# 创建参数
theta = Parameter('theta')
phi = Parameter('phi')

# 创建参数化电路
qc = Circuit(2)
qc.rx(theta, 0)
qc.ry(phi, 1)
qc.rz(theta, 0)  # 同一参数可多次使用

# 检查参数
print(qc.parameters)        # {Parameter(theta), Parameter(phi)}
print(qc.is_parameterized())  # True

# 绑定参数 (两种方法等价)
bound_qc = qc.bind_parameters({theta: np.pi/2, phi: np.pi/4})
# 或
bound_qc = qc.assign_parameters({theta: np.pi/2, phi: np.pi/4})

print(bound_qc.is_parameterized())  # False

# 部分绑定
partial_qc = qc.bind_parameters({theta: np.pi/2})
print(partial_qc.parameters)  # {Parameter(phi)}

# 原地修改
qc.bind_parameters({theta: np.pi/2}, inplace=True)
```

## 电路操作

### 基本操作

```python
# 复制
qc_copy = qc.copy()

# 组合电路
qc1 = Circuit(2)
qc1.h(0)
qc2 = Circuit(2)
qc2.cx(0, 1)
qc1.compose(qc2)  # qc1 现在包含 h + cx

# 电路逆 (不含测量/重置)
qc_inv = qc.inverse()

# 电路属性
print(qc.n_qubits)           # 量子比特数
print(qc.n_clbits)           # 经典比特数
print(qc.depth)              # 电路深度
print(qc.num_nonlocal_gates) # 多比特门数量
```

### 导出格式

```python
# Janus 字典格式
qc.to_dict_list()
# [{'name': 'h', 'qubits': [0], 'params': []}, ...]

# 元组格式 (Qiskit 兼容)
qc.to_tuple_list()
# [('h', [0], []), ('cx', [0, 1], []), ...]

# 指令列表转换
from janus.circuit.converters import to_instruction_list, from_instruction_list

inst_list = to_instruction_list(qc)
qc2 = from_instruction_list(inst_list, n_qubits=2)
```

## 编译器

### 基础优化

```python
from janus.compiler import compile_circuit

qc = Circuit(2)
qc.h(0)
qc.h(0)  # 冗余，会被消除
qc.rz(np.pi/4, 0)
qc.rz(np.pi/4, 0)  # 会被合并

optimized = compile_circuit(qc, optimization_level=2)
```

### 优化级别

| 级别 | 优化内容 |
|-----|---------|
| 0 | 无优化 |
| 1 | 移除恒等门、消除逆门对 (X-X, H-H 等) |
| 2 | 级别1 + 合并连续旋转门 (RZ+RZ → RZ) |

### 自定义 Pass

```python
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
```

## 编码器

### Schmidt 编码

```python
from janus.encode.schmidt_encode import schmidt_encode
import numpy as np

# 准备归一化的量子态
data = [1/np.sqrt(2), 1/np.sqrt(2), 0, 0]

# 编码为量子电路
circuit = schmidt_encode(q_size=4, data=data, cutoff=1e-4)
print(f"编码电路: {len(circuit.instructions)} 门")
```

## 模块结构

```
janus/
├── circuit/
│   ├── circuit.py      # 核心 Circuit 类
│   ├── gate.py         # 门基类
│   ├── instruction.py  # 指令类
│   ├── layer.py        # 层表示
│   ├── dag.py          # DAG 表示 (DAGCircuit, DAGDependency, BlockCollector 等)
│   ├── converters.py   # 格式转换
│   ├── parameter.py    # 参数化支持 (Parameter, ParameterExpression)
│   ├── qubit.py        # 量子比特和寄存器
│   ├── clbit.py        # 经典比特和寄存器
│   └── library/        # 标准门库 (60+ 门)
│       ├── __init__.py
│       └── standard_gates.py
├── compiler/
│   ├── compiler.py     # 编译主函数
│   └── passes.py       # 优化 Pass
└── encode/
    └── schmidt_encode.py  # Schmidt 编码
```



## 许可证

MIT License
