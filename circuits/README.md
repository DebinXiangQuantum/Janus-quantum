# 电路 JSON 文件存储目录

此目录用于存放量子电路的 JSON 文件。

## 文件格式

分层格式，每层是一个门列表：

```json
[
  [{"name": "h", "qubits": [0], "params": []}],
  [{"name": "cx", "qubits": [0, 1], "params": []}],
  [{"name": "rx", "qubits": [0], "params": [1.57]}]
]
```

## 使用方法

```python
from janus.circuit import load_circuit, list_circuits

# 列出所有电路文件
print(list_circuits())

# 加载电路
qc = load_circuit(name='bell')

# 或指定完整路径
qc = load_circuit(filepath='./my_circuit.json')
```

## 命令行工具

```bash
# 查看电路信息
python -m janus.circuit.cli info janus/circuit/circuits/bell.json

# 绘制电路
python -m janus.circuit.cli draw janus/circuit/circuits/bell.json
```
