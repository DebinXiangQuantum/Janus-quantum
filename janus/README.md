# Janus 量子电路编译器

Janus 是一个轻量级的量子电路描述和编译框架，提供简洁的 API 来构建、操作和分析量子电路。

## 安装

```bash
# 确保在项目根目录
cd your-project
```

## 快速开始

### 创建电路

```python
from janus.circuit import Circuit

# 创建一个 2 量子比特的电路
qc = Circuit(2, name="my_circuit")

# 添加量子门
qc.h(0)           # Hadamard 门
qc.cx(0, 1)       # CNOT 门

print(qc)
```

### 创建 Bell 态

```python
from janus.circuit import Circuit

qc = Circuit(2, name="Bell")
qc.h(0)
qc.cx(0, 1)

print(f"深度: {qc.depth}")
print(f"门数: {qc.n_gates}")
print(qc.draw())
```

输出：
```
q0: ─[h]──●─
q1: ────X─
```

## 支持的量子门

### 单比特门

| 方法 | 门 | 描述 |
|------|-----|------|
| `qc.h(qubit)` | H | Hadamard 门 |
| `qc.x(qubit)` | X | Pauli-X 门 (NOT) |
| `qc.y(qubit)` | Y | Pauli-Y 门 |
| `qc.z(qubit)` | Z | Pauli-Z 门 |
| `qc.s(qubit)` | S | S 门 (√Z) |
| `qc.t(qubit)` | T | T 门 (√S) |

### 参数化单比特门

| 方法 | 门 | 描述 |
|------|-----|------|
| `qc.rx(theta, qubit)` | RX(θ) | 绕 X 轴旋转 |
| `qc.ry(theta, qubit)` | RY(θ) | 绕 Y 轴旋转 |
| `qc.rz(theta, qubit)` | RZ(θ) | 绕 Z 轴旋转 |
| `qc.u(θ, φ, λ, qubit)` | U(θ,φ,λ) | 通用单比特门 |

### 两比特门

| 方法 | 门 | 描述 |
|------|-----|------|
| `qc.cx(ctrl, tgt)` | CX | CNOT 门 |
| `qc.cz(ctrl, tgt)` | CZ | 受控 Z 门 |
| `qc.crz(theta, ctrl, tgt)` | CRZ(θ) | 受控 RZ 门 |
| `qc.swap(q1, q2)` | SWAP | 交换门 |

## 电路属性

```python
qc = Circuit(4)
qc.h(0)
qc.cx(0, 1)
qc.cx(1, 2)

# 基本属性
qc.n_qubits           # 量子比特数: 4
qc.n_gates            # 门总数: 3
qc.depth              # 电路深度: 3
qc.num_two_qubit_gates  # 两比特门数: 2

# 获取指令
qc.instructions       # 所有指令列表
qc.layers             # 分层表示

# 获取操作的量子比特
qc.operated_qubits    # [0, 1, 2]
```

## 分层表示

Janus 自动计算电路的分层结构，同一层的门可以并行执行：

```python
qc = Circuit(4)
qc.h(0)
qc.h(1)
qc.h(2)
qc.h(3)
qc.cx(0, 1)
qc.cx(2, 3)
qc.cx(1, 2)

for i, layer in enumerate(qc.layers):
    gates = [inst.name for inst in layer]
    print(f"Layer {i}: {gates}")
```

输出：
```
Layer 0: ['h', 'h', 'h', 'h']
Layer 1: ['cx', 'cx']
Layer 2: ['cx']
```

## 电路操作

### 复制电路

```python
qc1 = Circuit(2)
qc1.h(0)

qc2 = qc1.copy()
qc2.cx(0, 1)

print(qc1.n_gates)  # 1
print(qc2.n_gates)  # 2
```

### 组合电路

```python
qc1 = Circuit(2)
qc1.h(0)

qc2 = Circuit(2)
qc2.cx(0, 1)

# 方法 1: compose
qc1.compose(qc2)

# 方法 2: 加法
qc3 = qc1 + qc2
```

### 电路求逆

```python
qc = Circuit(2)
qc.rx(np.pi/2, 0)
qc.cx(0, 1)

qc_inv = qc.inverse()  # 逆电路
```

## 门矩阵

获取门的酉矩阵表示：

```python
from janus.circuit import HGate, CXGate
import numpy as np

h = HGate()
print(h.to_matrix())
# [[0.707+0.j  0.707+0.j]
#  [0.707+0.j -0.707+0.j]]

cx = CXGate()
print(cx.to_matrix())
# [[1 0 0 0]
#  [0 1 0 0]
#  [0 0 0 1]
#  [0 0 1 0]]
```

## 格式转换

### 转换为字典格式

```python
qc = Circuit(2)
qc.h(0)
qc.cx(0, 1)

# 指令列表
qc.to_instructions()
# [{'name': 'h', 'qubits': [0], 'params': []},
#  {'name': 'cx', 'qubits': [0, 1], 'params': []}]

# 分层格式
qc.to_layers()
# [[{'name': 'h', 'qubits': [0], 'params': []}],
#  [{'name': 'cx', 'qubits': [0, 1], 'params': []}]]
```

### 从字典创建

```python
layers = [
    [{'name': 'h', 'qubits': [0], 'params': []}],
    [{'name': 'cx', 'qubits': [0, 1], 'params': []}]
]
qc = Circuit.from_layers(layers, n_qubits=2)
```

## Qiskit 互转

```python
from janus.circuit import Circuit
from janus.circuit.converters import to_qiskit, from_qiskit

# Janus -> Qiskit
jc = Circuit(2)
jc.h(0)
jc.cx(0, 1)
qiskit_circuit = to_qiskit(jc)

# Qiskit -> Janus
from qiskit import QuantumCircuit
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
janus_circuit = from_qiskit(qc)
```

## 直接使用门类

```python
from janus.circuit import Circuit, HGate, RXGate, CXGate
import numpy as np

qc = Circuit(2)

# 使用 append 添加门
qc.append(HGate(), [0])
qc.append(RXGate(np.pi/2), [1])
qc.append(CXGate(), [0, 1])
```

## 项目结构

```
janus/
├── circuit/
│   ├── __init__.py          # 模块入口
│   ├── operation.py         # 操作基类
│   ├── gate.py              # 量子门基类
│   ├── instruction.py       # 电路指令
│   ├── layer.py             # 电路层
│   ├── circuit.py           # 核心电路类
│   ├── qubit.py             # 量子比特
│   ├── converters.py        # 格式转换
│   └── library/
│       ├── __init__.py
│       └── standard_gates.py  # 标准门库
└── compiler/                # 编译器 (开发中)
```

## 示例：量子傅里叶变换 (QFT)

```python
from janus.circuit import Circuit
import numpy as np

def qft(n_qubits):
    qc = Circuit(n_qubits, name=f"QFT_{n_qubits}")
    
    for i in range(n_qubits):
        qc.h(i)
        for j in range(i + 1, n_qubits):
            angle = np.pi / (2 ** (j - i))
            qc.crz(angle, j, i)
    
    # SWAP
    for i in range(n_qubits // 2):
        qc.swap(i, n_qubits - 1 - i)
    
    return qc

qc = qft(3)
print(f"QFT depth: {qc.depth}")
print(f"QFT gates: {qc.n_gates}")
```

## License

MIT License
