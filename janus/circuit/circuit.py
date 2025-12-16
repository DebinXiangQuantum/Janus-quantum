"""
Janus 量子电路

核心电路类，提供量子电路的构建和操作
"""
from typing import List, Optional, Union, Iterator
import uuid
import copy
import numpy as np

from .gate import Gate
from .instruction import Instruction
from .layer import Layer
from .qubit import Qubit, QuantumRegister


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
        name: Optional[str] = None
    ):
        """
        初始化量子电路
        
        Args:
            n_qubits: 量子比特数量
            name: 电路名称（可选）
        """
        self._id = uuid.uuid4()
        self._n_qubits = n_qubits
        self._name = name
        
        # 指令列表（顺序存储）
        self._instructions: List[Instruction] = []
        
        # 分层存储（延迟计算）
        self._layers: Optional[List[Layer]] = None
        self._layers_dirty = True
        
        # 量子寄存器
        self._qreg = QuantumRegister(n_qubits, "q")
    
    @property
    def id(self) -> uuid.UUID:
        return self._id
    
    @property
    def n_qubits(self) -> int:
        return self._n_qubits
    
    @property
    def num_qubits(self) -> int:
        """兼容 qiskit 命名"""
        return self._n_qubits
    
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
        """兼容 qiskit 命名"""
        return self._instructions
    
    @property
    def depth(self) -> int:
        """获取电路深度（层数）"""
        return len(self.layers)
    
    @property
    def n_gates(self) -> int:
        """获取门的总数"""
        return len(self._instructions)
    
    @property
    def num_two_qubit_gates(self) -> int:
        """获取两比特门的数量"""
        return sum(1 for inst in self._instructions if len(inst.qubits) == 2)
    
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
    
    # ==================== 标准门方法 ====================
    
    def h(self, qubit: int) -> 'Circuit':
        """添加 Hadamard 门"""
        from .library.standard_gates import HGate
        return self._add_gate(HGate(), [qubit])
    
    def x(self, qubit: int) -> 'Circuit':
        """添加 Pauli-X 门"""
        from .library.standard_gates import XGate
        return self._add_gate(XGate(), [qubit])
    
    def y(self, qubit: int) -> 'Circuit':
        """添加 Pauli-Y 门"""
        from .library.standard_gates import YGate
        return self._add_gate(YGate(), [qubit])
    
    def z(self, qubit: int) -> 'Circuit':
        """添加 Pauli-Z 门"""
        from .library.standard_gates import ZGate
        return self._add_gate(ZGate(), [qubit])
    
    def s(self, qubit: int) -> 'Circuit':
        """添加 S 门 (sqrt(Z))"""
        from .library.standard_gates import SGate
        return self._add_gate(SGate(), [qubit])
    
    def t(self, qubit: int) -> 'Circuit':
        """添加 T 门 (sqrt(S))"""
        from .library.standard_gates import TGate
        return self._add_gate(TGate(), [qubit])
    
    def rx(self, theta: float, qubit: int) -> 'Circuit':
        """添加 RX 旋转门"""
        from .library.standard_gates import RXGate
        return self._add_gate(RXGate(theta), [qubit])
    
    def ry(self, theta: float, qubit: int) -> 'Circuit':
        """添加 RY 旋转门"""
        from .library.standard_gates import RYGate
        return self._add_gate(RYGate(theta), [qubit])
    
    def rz(self, theta: float, qubit: int) -> 'Circuit':
        """添加 RZ 旋转门"""
        from .library.standard_gates import RZGate
        return self._add_gate(RZGate(theta), [qubit])
    
    def u(self, theta: float, phi: float, lam: float, qubit: int) -> 'Circuit':
        """添加 U 门（通用单比特门）"""
        from .library.standard_gates import UGate
        return self._add_gate(UGate(theta, phi, lam), [qubit])
    
    def cx(self, control: int, target: int) -> 'Circuit':
        """添加 CNOT (CX) 门"""
        from .library.standard_gates import CXGate
        return self._add_gate(CXGate(), [control, target])
    
    def cz(self, control: int, target: int) -> 'Circuit':
        """添加 CZ 门"""
        from .library.standard_gates import CZGate
        return self._add_gate(CZGate(), [control, target])
    
    def crz(self, theta: float, control: int, target: int) -> 'Circuit':
        """添加 CRZ 门"""
        from .library.standard_gates import CRZGate
        return self._add_gate(CRZGate(theta), [control, target])
    
    def swap(self, qubit1: int, qubit2: int) -> 'Circuit':
        """添加 SWAP 门"""
        from .library.standard_gates import SwapGate
        return self._add_gate(SwapGate(), [qubit1, qubit2])
    
    def barrier(self, qubits: Optional[List[int]] = None) -> 'Circuit':
        """添加 barrier（用于分隔电路层）"""
        if qubits is None:
            qubits = list(range(self._n_qubits))
        from .library.standard_gates import Barrier
        return self._add_gate(Barrier(len(qubits)), qubits)
    
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
        new_circuit = Circuit(self._n_qubits, self._name)
        new_circuit._instructions = [inst.copy() for inst in self._instructions]
        new_circuit._layers_dirty = True
        return new_circuit
    
    def compose(self, other: 'Circuit', qubits: Optional[List[int]] = None) -> 'Circuit':
        """
        将另一个电路组合到当前电路
        
        Args:
            other: 要组合的电路
            qubits: 映射的量子比特（可选）
        """
        if qubits is None:
            qubits = list(range(other.n_qubits))
        
        if len(qubits) != other.n_qubits:
            raise ValueError("Qubit mapping size mismatch")
        
        for inst in other.instructions:
            mapped_qubits = [qubits[q] for q in inst.qubits]
            self.append(inst.operation.copy(), mapped_qubits, inst.clbits)
        
        return self
    
    def __add__(self, other: 'Circuit') -> 'Circuit':
        """电路连接"""
        n_qubits = max(self._n_qubits, other._n_qubits)
        new_circuit = Circuit(n_qubits)
        new_circuit.compose(self)
        new_circuit.compose(other)
        return new_circuit
    
    def inverse(self) -> 'Circuit':
        """返回电路的逆"""
        new_circuit = Circuit(self._n_qubits, f"{self._name}_inv" if self._name else None)
        for inst in reversed(self._instructions):
            inv_gate = inst.operation.inverse()
            new_circuit.append(inv_gate, inst.qubits, inst.clbits)
        return new_circuit
    
    # ==================== 转换方法 ====================
    
    def to_layers(self) -> List[List[dict]]:
        """转换为分层的字典列表格式（兼容旧格式）"""
        return [layer.to_list() for layer in self.layers]
    
    def to_instructions(self) -> List[dict]:
        """转换为指令字典列表"""
        return [inst.to_dict() for inst in self._instructions]
    
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
