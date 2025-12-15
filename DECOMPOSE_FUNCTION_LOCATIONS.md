# 量子门分解函数位置记录

## 单量子比特门分解函数

### `decompose_one_qubit`
- **定义位置**：`qiskit/synthesis/one_qubit/one_qubit_decompose.py`
- **导入方式**：`from qiskit.synthesis.one_qubit import decompose_one_qubit`
- **功能**：将任意单量子比特门分解为指定基下的门序列
- **参数**：
  - `gate`：输入的量子门（支持Gate、Operator、np.ndarray类型）
  - `basis`：分解基，支持'ZYZ', 'ZXZ', 'XYX', 'U3', 'U' (默认: 'U3')
  - `simplify`：是否简化结果电路 (默认: True)
  - `use_dag_circuit`：是否返回DAGCircuit对象 (默认: False)
- **返回值**：分解后的QuantumCircuit或DAGCircuit对象

## 双量子比特门转换接口函数

### `decompose_two_qubit_gate`
- **定义位置**：`qiskit/synthesis/two_qubit/two_qubit_decompose.py`
- **导入方式**：`from qiskit.synthesis.two_qubit import decompose_two_qubit_gate`
- **功能**：将双量子比特门转换为指定基门序列
- **参数**：
  - `gate`：输入的双量子比特门（支持Gate、Operator、np.ndarray类型）
  - `basis`：目标基门，支持'cx', 'cz', 'iswap', 'rxx', 'ryy', 'rzz'等，或自定义双量子比特门 (默认: 'cx')
  - `use_dag_circuit`：是否返回DAGCircuit对象 (默认: False)
  - `synthesis_options`：分解算法的额外选项 (默认: {})
- **返回值**：转换后的QuantumCircuit或DAGCircuit对象

## 多控制Toffoli门分解统一接口函数

### `decompose_multi_control_toffoli`
- **定义位置**：`qiskit/synthesis/multi_controlled/mcx_synthesis.py`
- **导入方式**：`from qiskit.synthesis.multi_controlled import decompose_multi_control_toffoli`
- **功能**：将多控制Toffoli门分解为指定的基础门序列，支持多种分解方法
- **参数**：
  - `num_ctrl_qubits`：控制量子比特数量
  - `num_target_qubits`：目标量子比特数量，默认为1
  - `num_ancilla_qubits`：辅助量子比特数量，默认为0
  - `method`：分解方法，可选值包括：
    - 'noaux'或'v24'：无辅助量子比特的方法（默认）
    - 'hp24'：无辅助量子比特的高效方法（线性门数）
    - 'gray'：格雷码方法
    - 'b95'：Barenco 1995年的方法
    - 'i15'：Iten 2015年的方法
    - 'm15'：Maslov 2015年的方法
    - 'kg24'：Khattar和Gidney 2024年的方法（支持1-2个辅助量子比特）
  - `ancilla_type`：辅助量子比特类型，可选'clean'或'dirty'，默认为'clean'
- **返回值**：包含分解后门序列的量子电路（QuantumCircuit类型）

## 受控门分解统一接口函数

### `decompose_controlled_gate`
- **定义位置**：`qiskit/circuit/library/standard_gates/controlled_gate_decompose.py`
- **导入方式**：`from qiskit.circuit.library.standard_gates import decompose_controlled_gate`
- **功能**：将各种受控门（如X、Y、Z、RX、RY、RZ等）分解为基本量子门，支持多种分解方法
- **参数**：
  - `gate`：要分解的基础门对象
  - `num_ctrl_qubits`：控制量子比特数量
  - `method`：分解方法，可选值为"noancilla"、"v24"、"hp24"、"gray"、"b95"、"i15"、"m15"、"kg24"等
  - `num_ancilla_qubits`：辅助量子比特数量
  - `mode`：辅助量子比特使用模式，可选值为"clean"（干净辅助量子比特）或"dirty"（脏辅助量子比特）
- **返回值**：分解后的量子电路


## 任意数量量子比特KAK分解统一接口函数

### `decompose_kak`
- **定义位置**：`qiskit/circuit/library/standard_gates/kak_decompose.py`
- **导入方式**：`from qiskit.circuit.library.standard_gates import decompose_kak`
- **功能**：使用KAK分解方法分解任意数量量子比特的门或幺正矩阵，返回量子电路。
  - 1量子比特：使用Euler角分解
  - 2量子比特：使用Cartan KAK分解（TwoQubitWeylDecomposition）
  - 3+量子比特：使用量子香农分解（QSD）
- **参数**：
  - `unitary_or_gate`：要分解的幺正矩阵或门对象
  - `fidelity`：目标保真度，默认为1.0 - 1.0e-9
  - `euler_basis`：用于单量子比特旋转的基，默认为'ZXZ'，有效值包括['ZXZ', 'ZYZ', 'XYX', 'XZX', 'U', 'U3', 'U321', 'U1X', 'PSX', 'ZSX', 'ZSXX', 'RR']
  - `simplify`：是否简化分解后的电路，默认值为False
  - `atol`：简化过程中检查零值的绝对容差，默认值为1e-12
- **返回值**：分解后的量子电路（QuantumCircuit对象）
-



## 电路到指令集转换接口函数

### `convert_circuit_to_instruction_set`
- **定义位置**：`circuit_to_instruction_set.py`
- **导入方式**：`from circuit_to_instruction_set import convert_circuit_to_instruction_set`
- **功能**：将任意量子电路转换为指定的指令集
- **参数**：
  - `circuit`：输入的量子电路对象
  - `instruction_set`：要转换到的基门名称列表（如 ['u3', 'cx']）
  - `coupling_map`：目标量子比特连接映射，若为None则不强制任何连接性约束
  - `optimization_level`：优化级别（0-3），级别越高生成的电路越优化但耗时越长
  - `seed_transpiler`：设置transpiler随机部分的种子
- **返回值**：使用指定指令集的转换后量子电路
- **使用示例**：
```python
from qiskit import QuantumCircuit
from circuit_to_instruction_set import convert_circuit_to_instruction_set

# 创建示例电路
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
qc.z(1)
qc.x(0)
qc.y(1)

# 转换为u3和cx门
instruction_set = ['u3', 'cx']
converted_qc = convert_circuit_to_instruction_set(qc, instruction_set)

# 转换为不同的指令集并应用耦合映射和高级优化
coupling_map = [[0, 1], [1, 2]]  # 3量子比特线性链
qc3 = QuantumCircuit(3)
qc3.h(0)
qc3.cx(0, 2)  # 非本地连接

converted_qc3 = convert_circuit_to_instruction_set(
    qc3, ['u', 'cx', 'rz'], coupling_map=coupling_map, optimization_level=2
)
```