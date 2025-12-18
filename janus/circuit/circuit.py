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
    
    def x(self, qubit: int, params: Optional[List] = None) -> 'Circuit':
        """添加 Pauli-X 门"""
        from .library.standard_gates import XGate
        gate = XGate()
        if params:
            gate.params = params
        return self._add_gate(gate, [qubit])
    
    def y(self, qubit: int, params: Optional[List] = None) -> 'Circuit':
        """添加 Pauli-Y 门"""
        from .library.standard_gates import YGate
        gate = YGate()
        if params:
            gate.params = params
        return self._add_gate(gate, [qubit])
    
    def z(self, qubit: int, params: Optional[List] = None) -> 'Circuit':
        """添加 Pauli-Z 门"""
        from .library.standard_gates import ZGate
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

    def mcry(self, theta: float, controls: List[int], target: int, params: Optional[List] = None) -> 'Circuit':
        """添加多控 RY 门（controls... -> target）"""
        from .library.standard_gates import MCRYGate
        if not controls:
            raise ValueError("mcry 需要至少 1 个控制比特")
        gate = MCRYGate(theta, num_controls=len(controls))
        if params:
            gate.params = params
        return self._add_gate(gate, controls + [target])
    
    def swap(self, qubit1: int, qubit2: int, params: Optional[List] = None) -> 'Circuit':
        """添加 SWAP 门"""
        from .library.standard_gates import SwapGate
        gate = SwapGate()
        if params:
            gate.params = params
        return self._add_gate(gate, [qubit1, qubit2])

    def cswap(self, control: int, qubit1: int, qubit2: int, params: Optional[List] = None) -> 'Circuit':
        """添加受控 SWAP（Fredkin）门：control, qubit1, qubit2"""
        from .library.standard_gates import CSWAPGate
        gate = CSWAPGate()
        if params:
            gate.params = params
        return self._add_gate(gate, [control, qubit1, qubit2])
    
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
        """绘制文本电路图（更接近“标准量子电路图”）
        
        - 每个量子比特使用 3 行（盒子上边/内容/下边），从而支持“竖线连到目标门顶部中心”
        - 支持多控制门：mcry（控制点 C/● + 竖线 + 目标 [ry] 盒子）
        - 支持 cswap：控制点 + 竖线 + 两个 swap 端点 x/×
        """
        import sys

        enc = (getattr(sys.stdout, "encoding", None) or "").lower()
        ascii_fallback = any(k in enc for k in ("gbk", "cp936"))

        # 字符集（在 gbk 下强制 ASCII，避免乱码）
        ch_wire = "-" if ascii_fallback else "─"
        ch_v = "|" if ascii_fallback else "│"
        ch_ctrl = "C" if ascii_fallback else "●"
        ch_swap = "x" if ascii_fallback else "×"

        # 盒子字符
        if ascii_fallback:
            bx_tl, bx_tr, bx_bl, bx_br = "+", "+", "+", "+"
            bx_h, bx_v = "-", "|"
            bx_mid_l, bx_mid_r = "|", "|"
            bx_top_conn = "+"  # 顶部中心连接点
        else:
            bx_tl, bx_tr, bx_bl, bx_br = "┌", "┐", "└", "┘"
            bx_h, bx_v = "─", "│"
            bx_mid_l, bx_mid_r = "┤", "├"
            bx_top_conn = "┬"

        cell_w =  nine = 9
        cell_w = 9  # 必须为奇数，方便定义“中心列”
        center = cell_w // 2
        box_w = 5  # 盒子宽度（必须为奇数）
        box_center = box_w // 2
        box_start = center - box_center
        box_end = box_start + box_w - 1

        def _blank_seg() -> list[str]:
            return [" "] * cell_w

        def _seg_wire_mid() -> list[str]:
            seg = [ch_wire] * cell_w
            return seg

        def _put(seg: list[str], col: int, s: str):
            for k, ch in enumerate(s):
                j = col + k
                if 0 <= j < len(seg):
                    seg[j] = ch

        def _draw_box(seg_top: list[str], seg_mid: list[str], seg_bot: list[str], label: str, controlled: bool):
            # 上边：┌──┬──┐（controlled）或 ┌─────┐
            seg_top[box_start] = bx_tl
            seg_top[box_end] = bx_tr
            for j in range(box_start + 1, box_end):
                seg_top[j] = bx_h
            if controlled:
                seg_top[center] = bx_top_conn

            # 中间：┤ ry ├
            seg_mid[box_start] = bx_mid_l
            seg_mid[box_end] = bx_mid_r
            for j in range(box_start + 1, box_end):
                seg_mid[j] = " "
            lbl = label[: (box_w - 2)]
            lbl_col = center - (len(lbl) // 2)
            _put(seg_mid, lbl_col, lbl)

            # 下边：└─────┘
            seg_bot[box_start] = bx_bl
            seg_bot[box_end] = bx_br
            for j in range(box_start + 1, box_end):
                seg_bot[j] = bx_h

        # 行数：每个 qubit 3 行
        nrows = self._n_qubits * 3
        # 前缀对齐
        label_mid = [f"q{i}: " for i in range(self._n_qubits)]
        prefix_w = max(len(s) for s in label_mid) if label_mid else 0
        rows = []
        for q in range(self._n_qubits):
            pad = " " * (prefix_w - len(label_mid[q]))
            rows.append(" " * prefix_w)               # top
            rows.append(label_mid[q] + pad)          # mid
            rows.append(" " * prefix_w)               # bot

        def r_top(q: int) -> int:
            return 3 * q

        def r_mid(q: int) -> int:
            return 3 * q + 1

        def r_bot(q: int) -> int:
            return 3 * q + 2

        for layer in self.layers:
            # 每个 qubit 三段：top 空白，mid 画 wire，bot 空白
            segs_top = [_blank_seg() for _ in range(self._n_qubits)]
            segs_mid = [_seg_wire_mid() for _ in range(self._n_qubits)]
            segs_bot = [_blank_seg() for _ in range(self._n_qubits)]

            def _draw_vertical_span(lo_q: int, hi_q: int):
                for q in range(lo_q, hi_q + 1):
                    # 允许在 wire 上画竖线（让连接“穿过”中间线）
                    segs_top[q][center] = ch_v if segs_top[q][center] == " " else segs_top[q][center]
                    segs_mid[q][center] = ch_v
                    segs_bot[q][center] = ch_v if segs_bot[q][center] == " " else segs_bot[q][center]

            for inst in layer:
                qs = list(inst.qubits)
                if not qs:
                    continue
                name = inst.name

                if name == "mcry":
                    # qubits = [controls..., target]
                    controls = qs[:-1]
                    target = qs[-1]
                    if controls:
                        lo, hi = min(controls + [target]), max(controls + [target])
                        _draw_vertical_span(lo, hi)
                        for c in controls:
                            segs_mid[c][center] = ch_ctrl
                        # 目标：画成受控 [ry] 盒子，并在顶部中心留连接点
                        _draw_box(segs_top[target], segs_mid[target], segs_bot[target], "ry", controlled=True)
                    else:
                        _draw_box(segs_top[target], segs_mid[target], segs_bot[target], "ry", controlled=False)

                elif name == "cswap":
                    # qubits = [control, q1, q2]
                    if len(qs) == 3:
                        c, a, b = qs
                        lo, hi = min(qs), max(qs)
                        _draw_vertical_span(lo, hi)
                        segs_mid[c][center] = ch_ctrl
                        segs_mid[a][center] = ch_swap
                        segs_mid[b][center] = ch_swap
                    else:
                        lo, hi = min(qs), max(qs)
                        _draw_vertical_span(lo, hi)
                        for q in qs:
                            _draw_box(segs_top[q], segs_mid[q], segs_bot[q], name[:2], controlled=False)

                elif name in ("cx", "cz", "crz"):
                    if len(qs) == 2:
                        c, t = qs
                        lo, hi = min(qs), max(qs)
                        _draw_vertical_span(lo, hi)
                        segs_mid[c][center] = ch_ctrl
                        _draw_box(segs_top[t], segs_mid[t], segs_bot[t], "x" if name == "cx" else "z" if name == "cz" else "rz", controlled=False)
                    else:
                        lo, hi = min(qs), max(qs)
                        _draw_vertical_span(lo, hi)
                        for q in qs:
                            _draw_box(segs_top[q], segs_mid[q], segs_bot[q], name[:2], controlled=False)

                elif name == "swap":
                    if len(qs) == 2:
                        a, b = qs
                        lo, hi = min(qs), max(qs)
                        _draw_vertical_span(lo, hi)
                        segs_mid[a][center] = ch_swap
                        segs_mid[b][center] = ch_swap
                    else:
                        lo, hi = min(qs), max(qs)
                        _draw_vertical_span(lo, hi)
                        for q in qs:
                            segs_mid[q][center] = ch_swap

                else:
                    # 单比特门：画盒子
                    if len(qs) == 1:
                        q = qs[0]
                        _draw_box(segs_top[q], segs_mid[q], segs_bot[q], name[:2], controlled=False)
                    else:
                        # 多比特未知门：先画竖线，再在参与 qubit 上画小盒子
                        lo, hi = min(qs), max(qs)
                        _draw_vertical_span(lo, hi)
                        for q in qs:
                            _draw_box(segs_top[q], segs_mid[q], segs_bot[q], name[:2], controlled=False)

            # 拼接这一层
            for q in range(self._n_qubits):
                rows[r_top(q)] += "".join(segs_top[q]) + " "
                rows[r_mid(q)] += "".join(segs_mid[q]) + " "
                rows[r_bot(q)] += "".join(segs_bot[q]) + " "

        return "\n".join(rows)
    
    def __len__(self) -> int:
        """返回层数"""
        return self.depth
    
    def __iter__(self) -> Iterator[Layer]:
        """迭代层"""
        return iter(self.layers)
    
    def __getitem__(self, index: int) -> Layer:
        """获取指定层"""
        return self.layers[index]
