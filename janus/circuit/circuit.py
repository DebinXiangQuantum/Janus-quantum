"""
Janus 量子电路

核心电路类，提供量子电路的构建和操作
"""
from typing import List, Optional, Union, Iterator, Dict, Set
import uuid
import copy
import numpy as np

from .gate import Gate
from .instruction import Instruction
from .layer import Layer
from .qubit import Qubit, QuantumRegister
from .clbit import Clbit, ClassicalRegister
from .parameter import Parameter, ParameterExpression, is_parameterized


class CallableProperty:
    """可调用的属性 - 允许既作为属性又作为方法使用"""
    def __init__(self, func):
        self.func = func

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        # 返回一个可以直接访问值或调用的对象
        value = self.func(obj)
        return CallableInt(value)


class CallableInt(int):
    """可调用的整数 - 支持 depth() 和 depth 两种用法"""
    def __call__(self):
        return int(self)


class Circuit:
    """
    量子电路类
    
    表示完整的量子电路，支持两种模式：
    1. 分层模式：电路由多个 Layer 组成，每层包含可并行执行的门
    2. 顺序模式：按添加顺序记录所有指令
    
    Attributes:
        n_qubits: 量子比特数
        name: 电路名称
    """
    
    def __init__(
        self,
        n_qubits: int,
        n_clbits: int = 0,
        name: Optional[str] = None
    ):
        """
        初始化量子电路
        
        Args:
            n_qubits: 量子比特数量
            n_clbits: 经典比特数量（默认为 0）
            name: 电路名称（可选）
        """
        self._id = uuid.uuid4()
        self._n_qubits = n_qubits
        self._n_clbits = n_clbits
        self._name = name
        
        # 指令列表（顺序存储）
        self._instructions: List[Instruction] = []
        
        # 分层存储（延迟计算）
        self._layers: Optional[List[Layer]] = None
        self._layers_dirty = True
        
        # 量子寄存器和经典寄存器
        self._qreg = QuantumRegister(n_qubits, "q")
        self._creg = ClassicalRegister(n_clbits, "c") if n_clbits > 0 else None
        
        # 参数追踪
        self._parameters: Set[Parameter] = set()
    
    @property
    def id(self) -> uuid.UUID:
        return self._id
    
    @property
    def n_qubits(self) -> int:
        return self._n_qubits
    
    @property
    def num_qubits(self) -> int:
        return self._n_qubits
    
    @property
    def n_clbits(self) -> int:
        return self._n_clbits
    
    @property
    def num_clbits(self) -> int:
        return self._n_clbits
    
    @property
    def clbits(self) -> List[Clbit]:
        """获取所有经典比特"""
        if self._creg:
            return list(self._creg)
        return []
    
    @property
    def parameters(self) -> Set[Parameter]:
        """获取电路中的所有参数"""
        return self._parameters.copy()
    
    @property
    def name(self) -> Optional[str]:
        return self._name
    
    @name.setter
    def name(self, value: str):
        self._name = value
    
    @property
    def instructions(self) -> List[Instruction]:
        """获取所有指令"""
        return self._instructions
    
    @property
    def data(self) -> List[Instruction]:
        return self._instructions

    @CallableProperty
    def depth(self) -> int:
        """获取电路深度（层数）- 支持 depth 和 depth() 两种用法"""
        return len(self.layers)

    @property
    def n_gates(self) -> int:
        """获取门的总数"""
        return len(self._instructions)

    @property
    def num_two_qubit_gates(self) -> int:
        """获取两比特门的数量"""
        return sum(1 for inst in self._instructions if len(inst.qubits) == 2)

    def count_ops(self) -> Dict[str, int]:
        """
        统计电路中各类操作的数量

        Returns:
            操作名称到数量的映射字典
        """
        counts = {}
        for inst in self._instructions:
            name = inst.operation.name if inst.operation else "unknown"
            counts[name] = counts.get(name, 0) + 1
        return counts

    @property
    def layers(self) -> List[Layer]:
        """获取分层表示（延迟计算）"""
        if self._layers_dirty or self._layers is None:
            self._compute_layers()
        return self._layers
    
    @property
    def qubits(self) -> List[Qubit]:
        """获取所有量子比特"""
        return list(self._qreg)
    
    @property
    def operated_qubits(self) -> List[int]:
        """获取实际被操作的量子比特"""
        qubits = set()
        for inst in self._instructions:
            qubits.update(inst.qubits)
        return sorted(list(qubits))
    
    # ==================== 添加门的方法 ====================
    
    def append(self, gate: Gate, qubits: List[int], clbits: Optional[List[int]] = None):
        """
        添加一个门到电路
        
        Args:
            gate: 要添加的门
            qubits: 作用的量子比特
            clbits: 作用的经典比特（可选）
        """
        self._validate_qubits(qubits)
        if clbits:
            self._validate_clbits(clbits)
        
        # 追踪参数
        for param in gate.params:
            if isinstance(param, Parameter):
                self._parameters.add(param)
            elif isinstance(param, ParameterExpression):
                self._parameters.update(param.parameters)
        
        instruction = Instruction(gate, qubits, clbits)
        self._instructions.append(instruction)
        self._layers_dirty = True
        return self
    
    def _add_gate(self, gate: Gate, qubits: List[int]) -> 'Circuit':
        """内部方法：添加门"""
        return self.append(gate, qubits)
    
    def _validate_qubits(self, qubits: List[int]):
        """验证量子比特索引"""
        for q in qubits:
            if q < 0 or q >= self._n_qubits:
                raise ValueError(f"Qubit index {q} out of range [0, {self._n_qubits})")
    
    def _validate_clbits(self, clbits: List[int]):
        """验证经典比特索引"""
        for c in clbits:
            if c < 0 or c >= self._n_clbits:
                raise ValueError(f"Clbit index {c} out of range [0, {self._n_clbits})")
    
    # ==================== 标准门方法 ====================
    
    def h(self, qubit: int, params: Optional[List] = None) -> 'Circuit':
        """添加 Hadamard 门"""
        from .library.standard_gates import HGate
        gate = HGate()
        if params:
            gate.params = params
        return self._add_gate(gate, [qubit])
    
    def x(self, qubit, params: Optional[List] = None) -> 'Circuit':
        """添加 Pauli-X 门

        Args:
            qubit: 量子比特索引或可迭代对象(如range)
            params: 可选参数
        """
        from .library.standard_gates import XGate

        # 如果qubit是可迭代对象,递归调用
        if hasattr(qubit, '__iter__') and not isinstance(qubit, (str, bytes)):
            for q in qubit:
                self.x(q, params)
            return self

        gate = XGate()
        if params:
            gate.params = params
        return self._add_gate(gate, [qubit])

    def y(self, qubit, params: Optional[List] = None) -> 'Circuit':
        """添加 Pauli-Y 门

        Args:
            qubit: 量子比特索引或可迭代对象(如range)
            params: 可选参数
        """
        from .library.standard_gates import YGate

        # 如果qubit是可迭代对象,递归调用
        if hasattr(qubit, '__iter__') and not isinstance(qubit, (str, bytes)):
            for q in qubit:
                self.y(q, params)
            return self

        gate = YGate()
        if params:
            gate.params = params
        return self._add_gate(gate, [qubit])

    def z(self, qubit, params: Optional[List] = None) -> 'Circuit':
        """添加 Pauli-Z 门

        Args:
            qubit: 量子比特索引或可迭代对象(如range)
            params: 可选参数
        """
        from .library.standard_gates import ZGate

        # 如果qubit是可迭代对象,递归调用
        if hasattr(qubit, '__iter__') and not isinstance(qubit, (str, bytes)):
            for q in qubit:
                self.z(q, params)
            return self

        gate = ZGate()
        if params:
            gate.params = params
        return self._add_gate(gate, [qubit])
    
    def s(self, qubit: int, params: Optional[List] = None) -> 'Circuit':
        """添加 S 门 (sqrt(Z))"""
        from .library.standard_gates import SGate
        gate = SGate()
        if params:
            gate.params = params
        return self._add_gate(gate, [qubit])
    
    def t(self, qubit: int, params: Optional[List] = None) -> 'Circuit':
        """添加 T 门 (sqrt(S))"""
        from .library.standard_gates import TGate
        gate = TGate()
        if params:
            gate.params = params
        return self._add_gate(gate, [qubit])
    
    def tdg(self, qubit: int, params: Optional[List] = None) -> 'Circuit':
        """添加 T† 门 (T 的共轭转置)"""
        from .library.standard_gates import TdgGate
        gate = TdgGate()
        if params:
            gate.params = params
        return self._add_gate(gate, [qubit])
    
    def sdg(self, qubit: int, params: Optional[List] = None) -> 'Circuit':
        """添加 S† 门 (S 的共轭转置)"""
        from .library.standard_gates import SdgGate
        gate = SdgGate()
        if params:
            gate.params = params
        return self._add_gate(gate, [qubit])
    
    def rx(self, theta: float, qubit: int, params: Optional[List] = None) -> 'Circuit':
        """添加 RX 旋转门"""
        from .library.standard_gates import RXGate
        gate = RXGate(theta)
        if params:
            gate.params = params
        return self._add_gate(gate, [qubit])
    
    def ry(self, theta: float, qubit: int, params: Optional[List] = None) -> 'Circuit':
        """添加 RY 旋转门"""
        from .library.standard_gates import RYGate
        gate = RYGate(theta)
        if params:
            gate.params = params
        return self._add_gate(gate, [qubit])
    
    def rz(self, theta: float, qubit: int, params: Optional[List] = None) -> 'Circuit':
        """添加 RZ 旋转门"""
        from .library.standard_gates import RZGate
        gate = RZGate(theta)
        if params:
            gate.params = params
        return self._add_gate(gate, [qubit])
    
    def u(self, theta: float, phi: float, lam: float, qubit: int, params: Optional[List] = None) -> 'Circuit':
        """添加 U 门（通用单比特门）"""
        from .library.standard_gates import UGate
        gate = UGate(theta, phi, lam)
        if params:
            gate.params = params
        return self._add_gate(gate, [qubit])
    
    def cx(self, control: int, target: int, params: Optional[List] = None) -> 'Circuit':
        """添加 CNOT (CX) 门"""
        from .library.standard_gates import CXGate
        gate = CXGate()
        if params:
            gate.params = params
        return self._add_gate(gate, [control, target])
    
    def cz(self, control: int, target: int, params: Optional[List] = None) -> 'Circuit':
        """添加 CZ 门"""
        from .library.standard_gates import CZGate
        gate = CZGate()
        if params:
            gate.params = params
        return self._add_gate(gate, [control, target])
    
    def crz(self, theta: float, control: int, target: int, params: Optional[List] = None) -> 'Circuit':
        """添加 CRZ 门"""
        from .library.standard_gates import CRZGate
        gate = CRZGate(theta)
        if params:
            gate.params = params
        return self._add_gate(gate, [control, target])
    
    def swap(self, qubit1: int, qubit2: int, params: Optional[List] = None) -> 'Circuit':
        """添加 SWAP 门"""
        from .library.standard_gates import SwapGate
        gate = SwapGate()
        if params:
            gate.params = params
        return self._add_gate(gate, [qubit1, qubit2])

    def mcx(self, control_qubits: List[int], target_qubit: int, params: Optional[List] = None) -> 'Circuit':
        """
        添加多控制X门 (MCX)

        Args:
            control_qubits: 控制量子比特列表
            target_qubit: 目标量子比特
            params: 可选参数列表

        Returns:
            self (用于链式调用)
        """
        num_ctrl = len(control_qubits)

        if num_ctrl == 0:
            # 无控制位,就是普通的X门
            return self.x(target_qubit)
        elif num_ctrl == 1:
            # 单控制位,就是CX门
            return self.cx(control_qubits[0], target_qubit)
        elif num_ctrl == 2:
            # 双控制位,就是Toffoli (CCX) 门
            from .library.standard_gates import CCXGate
            gate = CCXGate()
            if params:
                gate.params = params
            return self._add_gate(gate, control_qubits + [target_qubit])
        else:
            # 多控制位 (3+ controls),创建通用MCX门
            # 这里创建一个简单的Gate对象来表示多控制X门
            from .gate import Gate

            # Create a general multi-controlled X gate
            gate = Gate(f'c{num_ctrl}x', num_ctrl + 1, params or [])
            qubits = control_qubits + [target_qubit]

            return self._add_gate(gate, qubits)

    def cp(self, theta: float, control: int, target: int, params: Optional[List] = None) -> 'Circuit':
        """添加 Controlled-Phase (CP) 门"""
        from .library.standard_gates import CPhaseGate
        gate = CPhaseGate(theta)
        if params:
            gate.params = params
        return self._add_gate(gate, [control, target])

    def p(self, theta: float, qubit: int, params: Optional[List] = None) -> 'Circuit':
        """添加 Phase (P) 门"""
        from .library.standard_gates import PhaseGate
        gate = PhaseGate(theta)
        if params:
            gate.params = params
        return self._add_gate(gate, [qubit])

    def barrier(self, qubits: Optional[List[int]] = None) -> 'Circuit':
        """添加 barrier（用于分隔电路层）"""
        if qubits is None:
            qubits = list(range(self._n_qubits))
        from .library.standard_gates import Barrier
        return self._add_gate(Barrier(len(qubits)), qubits)
    
    def measure(self, qubit: int, clbit: int) -> 'Circuit':
        """
        添加测量操作
        
        Args:
            qubit: 要测量的量子比特
            clbit: 存储结果的经典比特
        """
        from .library.standard_gates import Measure
        return self.append(Measure(), [qubit], [clbit])
    
    def measure_all(self) -> 'Circuit':
        """测量所有量子比特到对应的经典比特"""
        if self._n_clbits < self._n_qubits:
            raise ValueError(f"Not enough classical bits. Need {self._n_qubits}, have {self._n_clbits}")
        for i in range(self._n_qubits):
            self.measure(i, i)
        return self
    
    # ==================== 分层计算 ====================
    
    def _compute_layers(self):
        """计算电路的分层表示"""
        self._layers = []
        qubit_last_layer = [-1] * self._n_qubits  # 每个量子比特最后出现的层
        
        for inst in self._instructions:
            # 跳过 barrier
            if inst.name == 'barrier':
                continue
            
            # 找到该指令应该放在哪一层
            min_layer = 0
            for q in inst.qubits:
                min_layer = max(min_layer, qubit_last_layer[q] + 1)
            
            # 确保有足够的层
            while len(self._layers) <= min_layer:
                self._layers.append(Layer(index=len(self._layers)))
            
            # 添加指令到对应层
            self._layers[min_layer].append(inst.copy())
            
            # 更新量子比特的最后层
            for q in inst.qubits:
                qubit_last_layer[q] = min_layer
        
        self._layers_dirty = False
    
    # ==================== 电路操作 ====================
    
    def copy(self) -> 'Circuit':
        """创建电路的深拷贝"""
        new_circuit = Circuit(self._n_qubits, self._n_clbits, self._name)
        new_circuit._instructions = [inst.copy() for inst in self._instructions]
        new_circuit._parameters = self._parameters.copy()
        new_circuit._layers_dirty = True
        return new_circuit
    
    def assign_parameters(
        self, 
        parameters: Dict[Parameter, float],
        inplace: bool = False
    ) -> 'Circuit':
        """
        为参数赋值
        
        Args:
            parameters: 参数到值的映射
            inplace: 是否原地修改
        
        Returns:
            赋值后的电路
        """
        if inplace:
            circuit = self
        else:
            circuit = self.copy()
        
        new_instructions = []
        for inst in circuit._instructions:
            new_params = []
            for param in inst.operation.params:
                if isinstance(param, Parameter):
                    if param in parameters:
                        new_params.append(parameters[param])
                    else:
                        new_params.append(param)
                elif isinstance(param, ParameterExpression):
                    bound = param.bind(parameters)
                    new_params.append(float(bound) if isinstance(bound, (int, float)) or bound.is_real() else bound)
                else:
                    new_params.append(param)
            
            # 创建新的门和指令
            new_gate = inst.operation.copy()
            new_gate.params = new_params
            new_inst = Instruction(new_gate, inst.qubits.copy(), inst.clbits.copy())
            new_instructions.append(new_inst)
        
        circuit._instructions = new_instructions
        
        # 更新参数集合
        circuit._parameters = set()
        for inst in circuit._instructions:
            for param in inst.operation.params:
                if isinstance(param, Parameter):
                    circuit._parameters.add(param)
                elif isinstance(param, ParameterExpression):
                    circuit._parameters.update(param.parameters)
        
        circuit._layers_dirty = True
        return circuit
    
    def is_parameterized(self) -> bool:
        """检查电路是否包含未绑定的参数"""
        return len(self._parameters) > 0
    
    def compose(self, other: 'Circuit', qubits: Optional[List[int]] = None, inplace: bool = False) -> 'Circuit':
        """
        将另一个电路组合到当前电路

        Args:
            other: 要组合的电路
            qubits: 映射的量子比特（可选）
            inplace: 是否原地修改（默认False）

        Returns:
            组合后的电路（如果inplace=True则返回self，否则返回新电路）
        """
        if qubits is None:
            qubits = list(range(other.n_qubits))

        if len(qubits) != other.n_qubits:
            raise ValueError("Qubit mapping size mismatch")

        # 确定操作的目标电路
        target = self if inplace else self.copy()

        for inst in other.instructions:
            mapped_qubits = [qubits[q] for q in inst.qubits]
            target.append(inst.operation.copy(), mapped_qubits, inst.clbits)

        return target
    
    def __add__(self, other: 'Circuit') -> 'Circuit':
        """电路连接"""
        n_qubits = max(self._n_qubits, other._n_qubits)
        new_circuit = Circuit(n_qubits)
        new_circuit.compose(self)
        new_circuit.compose(other)
        return new_circuit

    def __and__(self, other: 'Circuit') -> 'Circuit':
        """电路组合运算符 (返回新电路)"""
        return self.compose(other, inplace=False)

    def __iand__(self, other: 'Circuit') -> 'Circuit':
        """电路就地组合运算符"""
        return self.compose(other, inplace=True)

    def inverse(self) -> 'Circuit':
        """返回电路的逆"""
        # 修复: 如果name是字符串,n_clbits参数会被错误传递
        if self._name:
            new_circuit = Circuit(self._n_qubits, self._n_clbits, name=f"{self._name}_inv")
        else:
            new_circuit = Circuit(self._n_qubits, self._n_clbits)

        for inst in reversed(self._instructions):
            inv_gate = inst.operation.inverse()
            new_circuit.append(inv_gate, inst.qubits, inst.clbits)
        return new_circuit
    
    # ==================== 转换方法 ====================
    
    def to_layers(self) -> List[List[dict]]:
        """转换为分层的字典列表格式（兼容旧格式）"""
        return [layer.to_list() for layer in self.layers]
    
    def to_instructions(self) -> List[dict]:
        """转换为指令字典列表 (Janus 格式)"""
        return [inst.to_dict() for inst in self._instructions]
    
    def to_dict_list(self) -> List[dict]:
        """
        转换为字典列表 (Janus 格式)
        
        Returns:
            [{'name': 'h', 'qubits': [0], 'params': []}, ...]
        """
        return self.to_instructions()
    
    def to_tuple_list(self) -> List[tuple]:
        """
        转换为元组列表 (Qiskit 风格格式)
        
        Returns:
            [('h', [0], []), ('cx', [0, 1], []), ...]
        """
        return [(inst.name, inst.qubits, inst.params) for inst in self._instructions]
    
    @classmethod
    def from_layers(cls, layers: List[List[dict]], n_qubits: Optional[int] = None) -> 'Circuit':
        """从分层字典格式创建电路"""
        # 推断量子比特数
        if n_qubits is None:
            max_qubit = 0
            for layer in layers:
                for gate_dict in layer:
                    max_qubit = max(max_qubit, max(gate_dict['qubits']))
            n_qubits = max_qubit + 1
        
        circuit = cls(n_qubits)
        for layer in layers:
            for gate_dict in layer:
                gate = Gate.from_dict(gate_dict)
                circuit.append(gate, gate_dict['qubits'])
        
        return circuit
    
    # ==================== 显示方法 ====================
    
    def __repr__(self) -> str:
        return f"Circuit(n_qubits={self._n_qubits}, n_gates={self.n_gates}, depth={self.depth})"
    
    def __str__(self) -> str:
        lines = [f"Circuit: {self._name or 'unnamed'} ({self._n_qubits} qubits)"]
        lines.append("-" * 40)
        for i, inst in enumerate(self._instructions):
            lines.append(f"  {i}: {inst.operation} on {inst.qubits}")
        return "\n".join(lines)
    
    def draw(self, output: str = 'text') -> str:
        """
        绘制电路
        
        Args:
            output: 输出格式 ('text' 或 'mpl')
        """
        if output == 'text':
            return self._draw_text()
        else:
            raise NotImplementedError(f"Output format '{output}' not implemented")
    
    def _draw_text(self) -> str:
        """简单的文本绘制"""
        lines = []
        for i in range(self._n_qubits):
            line = f"q{i}: "
            for layer in self.layers:
                gate_str = "─"
                for inst in layer:
                    if i in inst.qubits:
                        if len(inst.qubits) == 1:
                            gate_str = f"[{inst.name}]"
                        else:
                            idx = inst.qubits.index(i)
                            if idx == 0:
                                gate_str = f"●" if inst.name in ('cx', 'cz', 'crz') else f"[{inst.name}]"
                            else:
                                gate_str = f"X" if inst.name == 'cx' else f"Z" if inst.name == 'cz' else f"[{inst.name}]"
                        break
                line += f"─{gate_str}─"
            lines.append(line)
        return "\n".join(lines)
    
    def __len__(self) -> int:
        """返回层数"""
        return self.depth
    
    def __iter__(self) -> Iterator[Layer]:
        """迭代层"""
        return iter(self.layers)
    
    def __getitem__(self, index: int) -> Layer:
        """获取指定层"""
        return self.layers[index]

    @staticmethod
    def _from_circuit_data(circuit_data, *, legacy_qubits=False, name=None):
        """
        从CircuitData创建Circuit对象

        Args:
            circuit_data: CircuitData对象或指令列表
            legacy_qubits: 是否使用旧版量子比特格式
            name: 电路名称(可选)

        Returns:
            Circuit对象
        """
        # 处理circuit_data
        if hasattr(circuit_data, 'qubits'):
            # CircuitData对象
            n_qubits = len(circuit_data.qubits)
            n_clbits = len(circuit_data.clbits) if hasattr(circuit_data, 'clbits') else 0
            instructions = list(circuit_data)
        else:
            # 简单的指令列表
            # 从指令中推断量子比特数
            max_qubit = 0
            max_clbit = 0

            # 检查数据格式
            normalized_data = []
            for item in circuit_data:
                if isinstance(item, (list, tuple)):
                    if len(item) == 2 and isinstance(item[0], int) and isinstance(item[1], int):
                        # [[control, target], ...] 格式 - 假设是CX门
                        control, target = item
                        max_qubit = max(max_qubit, control, target)
                        from circuit.library import CXGate
                        normalized_data.append((CXGate(), [control, target], []))
                    elif len(item) == 3:
                        # (inst, qargs, cargs) 格式
                        inst, qargs, cargs = item
                        if qargs:
                            max_qubit = max(max_qubit, max(qargs) if isinstance(qargs[0], int) else max(q._index for q in qargs))
                        if cargs:
                            max_clbit = max(max_clbit, max(cargs) if isinstance(cargs[0], int) else max(c._index for c in cargs))
                        normalized_data.append(item)
                    elif len(item) == 2:
                        # (inst, qargs) 格式
                        inst, qargs = item
                        if qargs:
                            max_qubit = max(max_qubit, max(qargs) if isinstance(qargs[0], int) else max(q._index for q in qargs))
                        normalized_data.append((inst, qargs, []))
                    else:
                        raise ValueError(f"Invalid circuit_data item: {item}")

            n_qubits = max_qubit + 1
            n_clbits = max_clbit + 1 if max_clbit > 0 else 0
            instructions = normalized_data

        # 创建电路
        circuit = Circuit(n_qubits, n_clbits)

        # 设置电路名称
        if name is not None:
            circuit.name = name

        # 添加指令
        for item in instructions:
            if isinstance(item, tuple):
                if len(item) == 3:
                    inst, qargs, cargs = item
                elif len(item) == 2:
                    inst, qargs = item
                    cargs = []
                else:
                    # 处理其他格式
                    circuit._instructions.append(item)
                    continue

                # 处理量子比特参数
                if qargs and not isinstance(qargs[0], int):
                    qargs = [q._index if hasattr(q, '_index') else q for q in qargs]
                if cargs and not isinstance(cargs[0], int):
                    cargs = [c._index if hasattr(c, '_index') else c for c in cargs]
                circuit.append(inst, qargs, cargs)
            else:
                # 处理其他格式
                circuit._instructions.append(item)

        return circuit