"""
Janus 量子电路

核心电路类，提供量子电路的构建和操作
"""
from typing import List, Optional, Union, Iterator, Dict, Set
import uuid
import copy
import numpy as np

from .gate import Gate, ControlledGate
from .instruction import Instruction
from .layer import Layer
from .qubit import Qubit, QuantumRegister
from .clbit import Clbit, ClassicalRegister
from .parameter import Parameter, ParameterExpression, is_parameterized


class GateBuilder:
    """
    门构建器，支持链式调用添加受控门
    
    Example:
        qc = Circuit(4)
        # 链式调用创建受控门
        qc.gate(U3Gate(np.pi/4, np.pi/4, np.pi/4), 3).control([0, 1, 2])
    """
    
    def __init__(self, circuit: 'Circuit', gate: Gate, target_qubits: List[int]):
        """
        初始化门构建器
        
        Args:
            circuit: 所属电路
            gate: 基础门
            target_qubits: 目标量子比特列表
        """
        self._circuit = circuit
        self._gate = gate
        self._target_qubits = target_qubits
        self._added = False
    
    def control(self, control_qubits: Union[int, List[int]]) -> 'Circuit':
        """
        添加控制比特，创建受控门
        
        Args:
            control_qubits: 控制比特索引或索引列表
        
        Returns:
            Circuit: 返回电路以支持继续链式调用
        
        Example:
            qc.gate(RXGate(np.pi/4), 2).control(0)        # 单控制 RX
            qc.gate(U3Gate(np.pi/4, 0, 0), 3).control([0, 1, 2])  # 三控制 U3
        """
        if isinstance(control_qubits, int):
            control_qubits = [control_qubits]
        
        num_ctrl = len(control_qubits)
        controlled_gate = self._gate.control(num_ctrl)
        
        # 量子比特顺序：控制比特 + 目标比特
        qubits = list(control_qubits) + self._target_qubits
        
        self._circuit.append(controlled_gate, qubits)
        self._added = True
        return self._circuit
    
    def add(self) -> 'Circuit':
        """
        直接添加门（不添加控制）
        
        Returns:
            Circuit: 返回电路以支持继续链式调用
        """
        self._circuit.append(self._gate, self._target_qubits)
        self._added = True
        return self._circuit
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        if not self._added:
            self._circuit.append(self._gate, self._target_qubits)


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
        """添加 Hadamard 门，可选参数"""
        from .library.standard_gates import HGate
        gate = HGate()
        if params:
            gate.params = params
        return self._add_gate(gate, [qubit])
    
    def x(self, qubit: int, params: Optional[List] = None) -> 'Circuit':
        """添加 Pauli-X 门，可选参数"""
        from .library.standard_gates import XGate
        gate = XGate()
        if params:
            gate.params = params
        return self._add_gate(gate, [qubit])
    
    def y(self, qubit: int, params: Optional[List] = None) -> 'Circuit':
        """添加 Pauli-Y 门，可选参数"""
        from .library.standard_gates import YGate
        gate = YGate()
        if params:
            gate.params = params
        return self._add_gate(gate, [qubit])
    
    def z(self, qubit: int, params: Optional[List] = None) -> 'Circuit':
        """添加 Pauli-Z 门，可选参数"""
        from .library.standard_gates import ZGate
        gate = ZGate()
        if params:
            gate.params = params
        return self._add_gate(gate, [qubit])
    
    def s(self, qubit: int, params: Optional[List] = None) -> 'Circuit':
        """添加 S 门 (sqrt(Z))，可选参数"""
        from .library.standard_gates import SGate
        gate = SGate()
        if params:
            gate.params = params
        return self._add_gate(gate, [qubit])
    
    def t(self, qubit: int, params: Optional[List] = None) -> 'Circuit':
        """添加 T 门 (sqrt(S))，可选参数"""
        from .library.standard_gates import TGate
        gate = TGate()
        if params:
            gate.params = params
        return self._add_gate(gate, [qubit])
    
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

    def cswap(self, control: int, qubit1: int, qubit2: int) -> 'Circuit':
        """添加受控 SWAP（Fredkin）门"""
        from .library.standard_gates import CSwapGate
        return self._add_gate(CSwapGate(), [control, qubit1, qubit2])
    
    # ==================== 扩展门方法 ====================
    
    def id(self, qubit: int) -> 'Circuit':
        """添加 Identity 门"""
        from .library.standard_gates import IGate
        return self._add_gate(IGate(), [qubit])
    
    def sdg(self, qubit: int) -> 'Circuit':
        """添加 S† 门"""
        from .library.standard_gates import SdgGate
        return self._add_gate(SdgGate(), [qubit])
    
    def tdg(self, qubit: int) -> 'Circuit':
        """添加 T† 门"""
        from .library.standard_gates import TdgGate
        return self._add_gate(TdgGate(), [qubit])
    
    def sx(self, qubit: int) -> 'Circuit':
        """添加 sqrt(X) 门"""
        from .library.standard_gates import SXGate
        return self._add_gate(SXGate(), [qubit])
    
    def sxdg(self, qubit: int) -> 'Circuit':
        """添加 sqrt(X)† 门"""
        from .library.standard_gates import SXdgGate
        return self._add_gate(SXdgGate(), [qubit])
    
    def p(self, lam: float, qubit: int) -> 'Circuit':
        """添加 Phase 门"""
        from .library.standard_gates import PhaseGate
        return self._add_gate(PhaseGate(lam), [qubit])
    
    def u1(self, lam: float, qubit: int) -> 'Circuit':
        """添加 U1 门"""
        from .library.standard_gates import U1Gate
        return self._add_gate(U1Gate(lam), [qubit])
    
    def u2(self, phi: float, lam: float, qubit: int) -> 'Circuit':
        """添加 U2 门"""
        from .library.standard_gates import U2Gate
        return self._add_gate(U2Gate(phi, lam), [qubit])
    
    def u3(self, theta: float, phi: float, lam: float, qubit: int) -> 'Circuit':
        """添加 U3 门"""
        from .library.standard_gates import U3Gate
        return self._add_gate(U3Gate(theta, phi, lam), [qubit])
    
    def r(self, theta: float, phi: float, qubit: int) -> 'Circuit':
        """添加 R 门"""
        from .library.standard_gates import RGate
        return self._add_gate(RGate(theta, phi), [qubit])
    
    def cy(self, control: int, target: int) -> 'Circuit':
        """添加 CY 门"""
        from .library.standard_gates import CYGate
        return self._add_gate(CYGate(), [control, target])
    
    def ch(self, control: int, target: int) -> 'Circuit':
        """添加 CH 门"""
        from .library.standard_gates import CHGate
        return self._add_gate(CHGate(), [control, target])
    
    def cs(self, control: int, target: int) -> 'Circuit':
        """添加 CS 门"""
        from .library.standard_gates import CSGate
        return self._add_gate(CSGate(), [control, target])
    
    def csdg(self, control: int, target: int) -> 'Circuit':
        """添加 CS† 门"""
        from .library.standard_gates import CSdgGate
        return self._add_gate(CSdgGate(), [control, target])
    
    def csx(self, control: int, target: int) -> 'Circuit':
        """添加 CSX 门"""
        from .library.standard_gates import CSXGate
        return self._add_gate(CSXGate(), [control, target])
    
    def dcx(self, qubit1: int, qubit2: int) -> 'Circuit':
        """添加 DCX 门"""
        from .library.standard_gates import DCXGate
        return self._add_gate(DCXGate(), [qubit1, qubit2])
    
    def ecr(self, qubit1: int, qubit2: int) -> 'Circuit':
        """添加 ECR 门"""
        from .library.standard_gates import ECRGate
        return self._add_gate(ECRGate(), [qubit1, qubit2])
    
    def iswap(self, qubit1: int, qubit2: int) -> 'Circuit':
        """添加 iSWAP 门"""
        from .library.standard_gates import iSwapGate
        return self._add_gate(iSwapGate(), [qubit1, qubit2])
    
    def crx(self, theta: float, control: int, target: int) -> 'Circuit':
        """添加 CRX 门"""
        from .library.standard_gates import CRXGate
        return self._add_gate(CRXGate(theta), [control, target])
    
    def cry(self, theta: float, control: int, target: int) -> 'Circuit':
        """添加 CRY 门"""
        from .library.standard_gates import CRYGate
        return self._add_gate(CRYGate(theta), [control, target])
    
    def cp(self, theta: float, control: int, target: int) -> 'Circuit':
        """添加 CPhase 门"""
        from .library.standard_gates import CPhaseGate
        return self._add_gate(CPhaseGate(theta), [control, target])
    
    def cu1(self, lam: float, control: int, target: int) -> 'Circuit':
        """添加 CU1 门"""
        from .library.standard_gates import CU1Gate
        return self._add_gate(CU1Gate(lam), [control, target])
    
    def cu3(self, theta: float, phi: float, lam: float, control: int, target: int) -> 'Circuit':
        """添加 CU3 门"""
        from .library.standard_gates import CU3Gate
        return self._add_gate(CU3Gate(theta, phi, lam), [control, target])
    
    def cu(self, theta: float, phi: float, lam: float, gamma: float, control: int, target: int) -> 'Circuit':
        """添加 CU 门"""
        from .library.standard_gates import CUGate
        return self._add_gate(CUGate(theta, phi, lam, gamma), [control, target])
    
    def rxx(self, theta: float, qubit1: int, qubit2: int) -> 'Circuit':
        """添加 RXX 门"""
        from .library.standard_gates import RXXGate
        return self._add_gate(RXXGate(theta), [qubit1, qubit2])
    
    def ryy(self, theta: float, qubit1: int, qubit2: int) -> 'Circuit':
        """添加 RYY 门"""
        from .library.standard_gates import RYYGate
        return self._add_gate(RYYGate(theta), [qubit1, qubit2])
    
    def rzz(self, theta: float, qubit1: int, qubit2: int) -> 'Circuit':
        """添加 RZZ 门"""
        from .library.standard_gates import RZZGate
        return self._add_gate(RZZGate(theta), [qubit1, qubit2])
    
    def rzx(self, theta: float, qubit1: int, qubit2: int) -> 'Circuit':
        """添加 RZX 门"""
        from .library.standard_gates import RZXGate
        return self._add_gate(RZXGate(theta), [qubit1, qubit2])
    
    def ccx(self, ctrl1: int, ctrl2: int, target: int) -> 'Circuit':
        """添加 CCX (Toffoli) 门"""
        from .library.standard_gates import CCXGate
        return self._add_gate(CCXGate(), [ctrl1, ctrl2, target])
    
    def ccz(self, ctrl1: int, ctrl2: int, target: int) -> 'Circuit':
        """添加 CCZ 门"""
        from .library.standard_gates import CCZGate
        return self._add_gate(CCZGate(), [ctrl1, ctrl2, target])
    
    def c3x(self, ctrl1: int, ctrl2: int, ctrl3: int, target: int) -> 'Circuit':
        """添加 C3X 门"""
        from .library.standard_gates import C3XGate
        return self._add_gate(C3XGate(), [ctrl1, ctrl2, ctrl3, target])
    
    def c4x(self, ctrl1: int, ctrl2: int, ctrl3: int, ctrl4: int, target: int) -> 'Circuit':
        """添加 C4X 门"""
        from .library.standard_gates import C4XGate
        return self._add_gate(C4XGate(), [ctrl1, ctrl2, ctrl3, ctrl4, target])

    def rccx(self, ctrl1: int, ctrl2: int, target: int) -> 'Circuit':
        """添加 RCCX (简化 Toffoli) 门"""
        from .library.standard_gates import RCCXGate
        return self._add_gate(RCCXGate(), [ctrl1, ctrl2, target])

    def rc3x(self, ctrl1: int, ctrl2: int, ctrl3: int, target: int) -> 'Circuit':
        """添加 RC3X (简化三控制 X) 门"""
        from .library.standard_gates import RC3XGate
        return self._add_gate(RC3XGate(), [ctrl1, ctrl2, ctrl3, target])

    def c3sx(self, ctrl1: int, ctrl2: int, ctrl3: int, target: int) -> 'Circuit':
        """添加 C3SX (三控制 √X) 门"""
        from .library.standard_gates import C3SXGate
        return self._add_gate(C3SXGate(), [ctrl1, ctrl2, ctrl3, target])

    # ==================== 特殊两比特门 ====================

    def xx_minus_yy(self, theta: float, beta: float, qubit1: int, qubit2: int) -> 'Circuit':
        """添加 XX-YY 相互作用门
        
        Args:
            theta: 旋转角度
            beta: 相位角度
            qubit1: 第一个量子比特
            qubit2: 第二个量子比特
        """
        from .library.standard_gates import XXMinusYYGate
        return self._add_gate(XXMinusYYGate(theta, beta), [qubit1, qubit2])

    def xx_plus_yy(self, theta: float, beta: float, qubit1: int, qubit2: int) -> 'Circuit':
        """添加 XX+YY 相互作用门
        
        Args:
            theta: 旋转角度
            beta: 相位角度
            qubit1: 第一个量子比特
            qubit2: 第二个量子比特
        """
        from .library.standard_gates import XXPlusYYGate
        return self._add_gate(XXPlusYYGate(theta, beta), [qubit1, qubit2])

    # ==================== 全局相位 ====================

    def global_phase(self, phase: float) -> 'Circuit':
        """添加全局相位门
        
        Args:
            phase: 相位角度
        """
        from .library.standard_gates import GlobalPhaseGate
        # 全局相位门不作用于任何量子比特
        return self.append(GlobalPhaseGate(phase), [])

    # ==================== 多控制门 ====================
    
    def mcx(self, controls: list, target: int) -> 'Circuit':
        """添加 MCX (多控制 X) 门
        
        Args:
            controls: 控制比特列表
            target: 目标比特
        """
        from .library.standard_gates import MCXGate
        qubits = list(controls) + [target]
        return self._add_gate(MCXGate(len(controls)), qubits)

    def mcx_gray(self, controls: list, target: int) -> 'Circuit':
        """添加 MCX (Gray code 实现) 门
        
        Args:
            controls: 控制比特列表
            target: 目标比特
        """
        from .library.standard_gates import MCXGrayCode
        qubits = list(controls) + [target]
        return self._add_gate(MCXGrayCode(len(controls)), qubits)

    def mcx_recursive(self, controls: list, target: int) -> 'Circuit':
        """添加 MCX (递归实现) 门
        
        Args:
            controls: 控制比特列表
            target: 目标比特
        """
        from .library.standard_gates import MCXRecursive
        qubits = list(controls) + [target]
        return self._add_gate(MCXRecursive(len(controls)), qubits)

    def mcx_vchain(self, controls: list, target: int) -> 'Circuit':
        """添加 MCX (V-chain 实现) 门
        
        Args:
            controls: 控制比特列表
            target: 目标比特
        """
        from .library.standard_gates import MCXVChain
        qubits = list(controls) + [target]
        return self._add_gate(MCXVChain(len(controls)), qubits)
    
    def mcp(self, theta: float, controls: list, target: int) -> 'Circuit':
        """添加 MCPhase (多控制 Phase) 门
        
        Args:
            theta: 相位角度
            controls: 控制比特列表
            target: 目标比特
        """
        from .library.standard_gates import MCPhaseGate
        qubits = list(controls) + [target]
        return self._add_gate(MCPhaseGate(theta, len(controls)), qubits)
    
    def mcu1(self, lam: float, controls: list, target: int) -> 'Circuit':
        """添加 MCU1 (多控制 U1) 门
        
        Args:
            lam: 相位角度
            controls: 控制比特列表
            target: 目标比特
        """
        from .library.standard_gates import MCU1Gate
        qubits = list(controls) + [target]
        return self._add_gate(MCU1Gate(lam, len(controls)), qubits)
    
    def mcrx(self, theta: float, controls: list, target: int) -> 'Circuit':
        """添加 MCRX (多控制 RX) 门
        
        Args:
            theta: 旋转角度
            controls: 控制比特列表
            target: 目标比特
        """
        from .library.standard_gates import MCRXGate
        qubits = list(controls) + [target]
        return self._add_gate(MCRXGate(theta, len(controls)), qubits)
    
    def mcry(self, theta: float, controls: list, target: int) -> 'Circuit':
        """添加 MCRY (多控制 RY) 门
        
        Args:
            theta: 旋转角度
            controls: 控制比特列表
            target: 目标比特
        """
        from .library.standard_gates import MCRYGate
        qubits = list(controls) + [target]
        return self._add_gate(MCRYGate(theta, len(controls)), qubits)
    
    def mcrz(self, theta: float, controls: list, target: int) -> 'Circuit':
        """添加 MCRZ (多控制 RZ) 门
        
        Args:
            theta: 旋转角度
            controls: 控制比特列表
            target: 目标比特
        """
        from .library.standard_gates import MCRZGate
        qubits = list(controls) + [target]
        return self._add_gate(MCRZGate(theta, len(controls)), qubits)
    
    def reset(self, qubit: int) -> 'Circuit':
        """添加 Reset 操作"""
        from .library.standard_gates import Reset
        return self._add_gate(Reset(), [qubit])
    
    def delay(self, duration: float, qubit: int, unit: str = 'dt') -> 'Circuit':
        """添加 Delay 操作"""
        from .library.standard_gates import Delay
        return self._add_gate(Delay(duration, unit), [qubit])
    
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

    # ==================== 链式调用支持 ====================

    def gate(self, gate: Gate, target_qubits: Union[int, List[int]]) -> 'GateBuilder':
        """
        添加门并返回 GateBuilder 以支持链式调用
        
        Args:
            gate: 要添加的门
            target_qubits: 目标量子比特（单个或列表）
        
        Returns:
            GateBuilder: 门构建器，可以调用 .control() 添加控制
        
        Example:
            from janus.circuit.library import U3Gate, RXGate
            
            qc = Circuit(4)
            
            # 创建受控 U3 门
            qc.gate(U3Gate(np.pi/4, np.pi/4, np.pi/4), 3).control([0, 1, 2])
            
            # 创建受控 RX 门
            qc.gate(RXGate(np.pi/2), 2).control(0)
            
            # 不调用 control 则作为普通门添加
            qc.gate(U3Gate(np.pi/4, 0, 0), 0)  # 普通 U3 门
        """
        if isinstance(target_qubits, int):
            target_qubits = [target_qubits]
        
        return GateBuilder(self, gate, target_qubits)

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
    
    def bind_parameters(
        self,
        parameters: Dict[Parameter, float],
        inplace: bool = False
    ) -> 'Circuit':
        """
        绑定参数值（assign_parameters 的别名）
        
        Args:
            parameters: 参数到值的映射
            inplace: 是否原地修改
        
        Returns:
            绑定参数后的电路
        
        Example:
            theta = Parameter('theta')
            qc = Circuit(2)
            qc.rx(theta, 0)
            qc.ry(theta, 1)
            
            # 绑定参数
            bound_qc = qc.bind_parameters({theta: np.pi/2})
        """
        return self.assign_parameters(parameters, inplace)
    
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
        new_name = f"{self._name}_inv" if self._name else None
        new_circuit = Circuit(self._n_qubits, self._n_clbits, new_name)
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
    
    def draw(self, output: str = 'text', filename: str = None, figsize: tuple = None, 
             dpi: int = 150, fold: int = None, line_length: int = None):
        """
        绘制电路
        
        Args:
            output: 输出格式 ('text', 'mpl', 'png')
            filename: 保存文件名（仅 'png' 模式）
            figsize: 图像大小 (width, height)，默认自动计算
            dpi: 图像分辨率，默认 150
            fold: 每行显示的最大层数（仅 'text' 模式），设为 -1 禁用折叠
            line_length: 每行最大字符数（仅 'text' 模式），默认自动检测终端宽度
        
        Returns:
            'text' 模式返回字符串，'mpl' 模式返回 Figure 对象
        """
        if output == 'text':
            return self._draw_text(fold=fold, line_length=line_length)
        elif output in ('mpl', 'png'):
            fig = self._draw_mpl(figsize=figsize)
            if output == 'png' and filename:
                fig.savefig(filename, dpi=dpi, bbox_inches='tight', facecolor='white')
                print(f"Circuit saved to {filename}")
            return fig
        else:
            raise NotImplementedError(f"Output format '{output}' not implemented")
    
    def _draw_text(self, fold: int = None, line_length: int = None) -> str:
        """绘制文本电路图（更接近“标准量子电路图”）
        
        - 每个量子比特使用 3 行（盒子上边/内容/下边），从而支持“竖线连到目标门顶部中心”
        - 支持多控制门：mcry（控制点 C/● + 竖线 + 目标 [ry] 盒子）
        - 支持 cswap：控制点 + 竖线 + 两个 swap 端点 x/×
        """
        import sys
        from shutil import get_terminal_size

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

        cell_w = 23  # 增加宽度以容纳参数
        center = cell_w // 2
        box_w = 19  # 盒子宽度增加以容纳参数
        box_center = box_w // 2
        box_start = center - box_center
        box_end = box_start + box_w - 1

        def _blank_seg() -> list[str]:
            return [" "] * cell_w

        def _seg_wire_mid() -> list[str]:
            seg = [ch_wire] * cell_w
            return seg

        def _pi_check(val: float, short: bool = False) -> str:
            """将数值转换为 π 表示形式（参考 Qiskit pi_check）
            
            Args:
                val: 数值
                short: 是否使用短格式
            
            Returns:
                str: π 表示或数值字符串
            """
            import math
            eps = 1e-6
            
            if abs(val) < eps:
                return "0"
            
            neg = "-" if val < 0 else ""
            abs_val = abs(val)
            
            # 检查是否是 π 的整数倍
            pi_mult = abs_val / math.pi
            if abs(pi_mult - round(pi_mult)) < eps:
                mult = int(round(pi_mult))
                if mult == 1:
                    return f"{neg}π"
                else:
                    return f"{neg}{mult}π"
            
            # 检查是否是 π/n 形式 (n = 2,3,4,5,6,8,12)
            # 短格式使用更紧凑的表示
            for denom in [2, 3, 4, 5, 6, 8, 12]:
                frac_val = math.pi / denom
                if abs(abs_val - frac_val) < eps:
                    if short and denom == 2:
                        return f"{neg}π/2"
                    return f"{neg}π/{denom}"
                # 检查 n*π/m 形式
                for numer in range(2, 16):
                    if abs(abs_val - numer * frac_val) < eps:
                        if short:
                            # 短格式：尝试约分或使用小数
                            from math import gcd
                            g = gcd(numer, denom)
                            n, d = numer // g, denom // g
                            if d == 1:
                                return f"{neg}{n}π"
                            return f"{neg}{n}π/{d}"
                        return f"{neg}{numer}π/{denom}"
            
            # 不是 π 的简单分数，返回数值
            if short:
                # 短格式：最多2位有效数字
                if abs_val >= 100:
                    return f"{int(round(val))}"
                elif abs_val >= 10:
                    return f"{val:.0f}"
                elif abs_val >= 1:
                    return f"{val:.1f}"
                else:
                    return f"{val:.2f}"
            else:
                # 标准格式：保留2位小数
                if abs_val >= 100:
                    return f"{int(round(val))}"
                elif abs_val >= 10:
                    return f"{val:.1f}"
                else:
                    return f"{val:.2f}"
        
        def _format_gate_label(name: str, params: list) -> str:
            """格式化门标签，包含参数"""
            if not params:
                return name
            
            max_label_len = box_w - 2  # 盒子内可用宽度
            
            # 第一遍：使用 π 表示（标准格式）
            param_strs = []
            for p in params:
                if isinstance(p, (int, float)):
                    param_strs.append(_pi_check(p, short=False))
                else:
                    param_strs.append(str(p)[:4])
            
            label = name + "(" + ",".join(param_strs) + ")"
            if len(label) <= max_label_len:
                return label
            
            # 第二遍：使用短格式
            param_strs = []
            for p in params:
                if isinstance(p, (int, float)):
                    param_strs.append(_pi_check(p, short=True))
                else:
                    param_strs.append(str(p)[:3])
            
            label = name + "(" + ",".join(param_strs) + ")"
            if len(label) <= max_label_len:
                return label
            
            # 第三遍：截断参数部分，保留括号
            params_str = ",".join(param_strs)
            available = max_label_len - len(name) - 2  # 减去 name 和 "()"
            if available > 0:
                return name + "(" + params_str[:available] + ")"
            else:
                return name[:max_label_len]

        def _put(seg: list[str], col: int, s: str):
            for k, ch in enumerate(s):
                j = col + k
                if 0 <= j < len(seg):
                    seg[j] = ch

        # 底部连接点字符
        bx_bot_conn = "+" if ascii_fallback else "┴"
        
        def _draw_box(seg_top: list[str], seg_mid: list[str], seg_bot: list[str], label: str, 
                      ctrl_above: bool = False, ctrl_below: bool = False):
            """绘制门的盒子
            
            Args:
                seg_top: 顶部行
                seg_mid: 中间行
                seg_bot: 底部行
                label: 门标签
                ctrl_above: 是否有控制点在上方（需要顶部连接点）
                ctrl_below: 是否有控制点在下方（需要底部连接点）
            """
            # 检查是否有竖线穿过（由其他门画的）
            has_line_above = seg_top[center] == ch_v
            has_line_below = seg_bot[center] == ch_v
            
            # 上边：┌──┬──┐（ctrl_above 或有竖线穿过）或 ┌─────┐
            seg_top[box_start] = bx_tl
            seg_top[box_end] = bx_tr
            for j in range(box_start + 1, box_end):
                seg_top[j] = bx_h
            if ctrl_above or has_line_above:
                seg_top[center] = bx_top_conn

            # 中间：┤ ry ├
            seg_mid[box_start] = bx_mid_l
            seg_mid[box_end] = bx_mid_r
            for j in range(box_start + 1, box_end):
                seg_mid[j] = " "
            lbl = label[: (box_w - 2)]
            lbl_col = center - (len(lbl) // 2)
            _put(seg_mid, lbl_col, lbl)

            # 下边：└──┴──┘（ctrl_below 或有竖线穿过）或 └─────┘
            seg_bot[box_start] = bx_bl
            seg_bot[box_end] = bx_br
            for j in range(box_start + 1, box_end):
                seg_bot[j] = bx_h
            if ctrl_below or has_line_below:
                seg_bot[center] = bx_bot_conn

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
            
            # 预先收集这一层中所有需要画盒子的量子比特位置
            box_qubits = set()
            for inst in layer:
                qs = list(inst.qubits)
                name = inst.name
                op = inst.operation
                
                # 确定哪些量子比特需要画盒子
                if isinstance(op, ControlledGate):
                    num_ctrl = op.ctrl_qubits
                    targets = qs[num_ctrl:]
                    box_qubits.update(targets)
                elif name in ("mcx", "mcx_gray", "mcx_recursive", "mcx_vchain", 
                            "mcp", "mcphase", "mcu1", "mcrx", "mcry", "mcrz"):
                    box_qubits.add(qs[-1])  # 目标
                elif name in ("cx", "cz", "crz", "crx", "cry", "cp", "cu", "cu1", "cu3", "ch", "cy"):
                    if len(qs) == 2:
                        box_qubits.add(qs[1])  # 目标
                elif name in ("ccx", "ccz", "c3x", "c4x", "rccx", "rc3x", "c3sx"):
                    box_qubits.add(qs[-1])  # 目标
                elif name not in ("swap", "cswap"):
                    # 其他门（单比特门等）
                    box_qubits.update(qs)

            def _draw_vertical_span(lo_q: int, hi_q: int, participating_qubits: set = None):
                """在 lo_q 和 hi_q 之间画竖线连接
                
                Args:
                    lo_q: 最上面的量子比特
                    hi_q: 最下面的量子比特
                    participating_qubits: 参与门操作的量子比特集合
                """
                if participating_qubits is None:
                    participating_qubits = set(range(lo_q, hi_q + 1))
                
                for q in range(lo_q, hi_q + 1):
                    is_participating = q in participating_qubits
                    is_top = q == lo_q
                    is_bottom = q == hi_q
                    
                    # 检查该位置是否有其他门的盒子（不是当前门的）
                    has_other_box = q in box_qubits and q not in participating_qubits
                    
                    if is_top and is_bottom:
                        # 只有一个量子比特，不需要画竖线
                        pass
                    elif is_top:
                        # 最上面的比特：只在下方画竖线
                        segs_bot[q][center] = ch_v
                    elif is_bottom:
                        # 最下面的比特：只在上方画竖线
                        segs_top[q][center] = ch_v
                    elif is_participating:
                        # 参与的中间比特：上下都画竖线，穿过 wire
                        segs_top[q][center] = ch_v
                        segs_mid[q][center] = ch_v
                        segs_bot[q][center] = ch_v
                    elif has_other_box:
                        # 有其他门的盒子：在盒子的顶部和底部画连接点
                        # 检查盒子是否已经画好（通过检查边框字符）
                        if segs_top[q][center] == bx_h:
                            segs_top[q][center] = bx_top_conn
                        elif segs_top[q][center] not in (bx_top_conn, ch_v):
                            # 如果不是连接点或竖线，画竖线
                            segs_top[q][center] = ch_v
                        # 否则保持原样（已经是连接点或竖线）
                        
                        if segs_bot[q][center] == bx_h:
                            segs_bot[q][center] = bx_bot_conn
                        elif segs_bot[q][center] not in (bx_bot_conn, ch_v):
                            # 如果不是连接点或竖线，画竖线
                            segs_bot[q][center] = ch_v
                        # 否则保持原样（已经是连接点或竖线）
                    else:
                        # 不参与的中间比特：竖线穿过，用交叉符号
                        segs_top[q][center] = ch_v
                        segs_mid[q][center] = "┼" if not ascii_fallback else "+"
                        segs_bot[q][center] = ch_v

            for inst in layer:
                qs = list(inst.qubits)
                if not qs:
                    continue
                name = inst.name
                op = inst.operation

                # 检查是否是 ControlledGate（通过 .control() 创建的）
                if isinstance(op, ControlledGate):
                    num_ctrl = op.ctrl_qubits
                    controls = qs[:num_ctrl]
                    targets = qs[num_ctrl:]
                    params = inst.params if hasattr(inst, 'params') else []
                    
                    # 获取基础门名称作为目标标签
                    base_name = op.base_gate.name
                    target_label = _format_gate_label(base_name, params)
                    
                    if controls and targets:
                        lo, hi = min(qs), max(qs)
                        _draw_vertical_span(lo, hi, set(qs))
                        for c in controls:
                            segs_mid[c][center] = ch_ctrl
                        # 目标门：根据控制点位置决定连接点
                        for t in targets:
                            has_ctrl_above = any(c < t for c in controls)
                            has_ctrl_below = any(c > t for c in controls)
                            _draw_box(segs_top[t], segs_mid[t], segs_bot[t], target_label, 
                                      ctrl_above=has_ctrl_above, ctrl_below=has_ctrl_below)
                    else:
                        for q in qs:
                            _draw_box(segs_top[q], segs_mid[q], segs_bot[q], target_label)

                elif name in ("mcx", "mcx_gray", "mcx_recursive", "mcx_vchain", 
                            "mcp", "mcphase", "mcu1", "mcrx", "mcry", "mcrz"):
                    # 多控制门: qubits = [controls..., target]
                    controls = qs[:-1]
                    target = qs[-1]
                    params = inst.params if hasattr(inst, 'params') else []
                    
                    # 确定目标门的标签
                    if name in ("mcx", "mcx_gray", "mcx_recursive", "mcx_vchain"):
                        target_label = "X"
                    elif name in ("mcp", "mcphase"):
                        target_label = _format_gate_label("p", params[:1] if params else [])
                    elif name == "mcu1":
                        target_label = _format_gate_label("u1", params[:1] if params else [])
                    elif name == "mcrx":
                        target_label = _format_gate_label("rx", params[:1] if params else [])
                    elif name == "mcry":
                        target_label = _format_gate_label("ry", params[:1] if params else [])
                    elif name == "mcrz":
                        target_label = _format_gate_label("rz", params[:1] if params else [])
                    else:
                        target_label = name
                    
                    if controls:
                        lo, hi = min(controls + [target]), max(controls + [target])
                        _draw_vertical_span(lo, hi, set(qs))
                        for c in controls:
                            segs_mid[c][center] = ch_ctrl
                        # 目标：根据控制点位置决定连接点
                        has_ctrl_above = any(c < target for c in controls)
                        has_ctrl_below = any(c > target for c in controls)
                        _draw_box(segs_top[target], segs_mid[target], segs_bot[target], target_label, 
                                  ctrl_above=has_ctrl_above, ctrl_below=has_ctrl_below)
                    else:
                        _draw_box(segs_top[target], segs_mid[target], segs_bot[target], target_label)

                elif name == "cswap":
                    # qubits = [control, q1, q2]
                    if len(qs) == 3:
                        c, a, b = qs
                        lo, hi = min(qs), max(qs)
                        _draw_vertical_span(lo, hi, set(qs))
                        segs_mid[c][center] = ch_ctrl
                        segs_mid[a][center] = ch_swap
                        segs_mid[b][center] = ch_swap
                    else:
                        lo, hi = min(qs), max(qs)
                        _draw_vertical_span(lo, hi, set(qs))
                        for q in qs:
                            _draw_box(segs_top[q], segs_mid[q], segs_bot[q], name[:2])

                elif name in ("cx", "cz", "crz", "crx", "cry", "cp", "cu", "cu1", "cu3", "ch", "cy"):
                    # 单控制门
                    if len(qs) == 2:
                        c, t = qs
                        lo, hi = min(qs), max(qs)
                        _draw_vertical_span(lo, hi, set(qs))
                        segs_mid[c][center] = ch_ctrl
                        # 目标门标签
                        params = inst.params if hasattr(inst, 'params') else []
                        if name == "cx":
                            target_label = "X"
                        elif name == "cz":
                            target_label = "Z"
                        elif name == "ch":
                            target_label = "H"
                        elif name == "cy":
                            target_label = "Y"
                        else:
                            # 受控旋转门，显示参数
                            base_name = name[1:] if name.startswith('c') else name
                            target_label = _format_gate_label(base_name, params)
                        # 根据控制点位置决定连接点
                        ctrl_above = c < t
                        ctrl_below = c > t
                        _draw_box(segs_top[t], segs_mid[t], segs_bot[t], target_label, 
                                  ctrl_above=ctrl_above, ctrl_below=ctrl_below)
                    else:
                        lo, hi = min(qs), max(qs)
                        _draw_vertical_span(lo, hi, set(qs))
                        for q in qs:
                            _draw_box(segs_top[q], segs_mid[q], segs_bot[q], name[:2])
                
                elif name in ("ccx", "ccz", "c3x", "c4x", "rccx", "rc3x", "c3sx"):
                    # 多控制门：前面都是控制，最后一个是目标
                    controls = qs[:-1]
                    target = qs[-1]
                    lo, hi = min(qs), max(qs)
                    _draw_vertical_span(lo, hi, set(qs))
                    for c in controls:
                        segs_mid[c][center] = ch_ctrl
                    # 目标门
                    if name in ("ccx", "c3x", "c4x", "rccx", "rc3x"):
                        target_label = "X"
                    elif name == "ccz":
                        target_label = "Z"
                    elif name == "c3sx":
                        target_label = "√X"
                    else:
                        target_label = name[-1].upper()
                    # 根据控制点位置决定连接点
                    has_ctrl_above = any(c < target for c in controls)
                    has_ctrl_below = any(c > target for c in controls)
                    _draw_box(segs_top[target], segs_mid[target], segs_bot[target], target_label, 
                              ctrl_above=has_ctrl_above, ctrl_below=has_ctrl_below)

                elif name == "swap":
                    if len(qs) == 2:
                        a, b = qs
                        lo, hi = min(qs), max(qs)
                        _draw_vertical_span(lo, hi, set(qs))
                        segs_mid[a][center] = ch_swap
                        segs_mid[b][center] = ch_swap
                    else:
                        lo, hi = min(qs), max(qs)
                        _draw_vertical_span(lo, hi, set(qs))
                        for q in qs:
                            segs_mid[q][center] = ch_swap

                else:
                    # 单比特门：画盒子，显示参数
                    params = inst.params if hasattr(inst, 'params') else []
                    label = _format_gate_label(name, params)
                    if len(qs) == 1:
                        q = qs[0]
                        _draw_box(segs_top[q], segs_mid[q], segs_bot[q], label)
                    else:
                        # 多比特未知门：先画竖线，再在参与 qubit 上画小盒子
                        lo, hi = min(qs), max(qs)
                        _draw_vertical_span(lo, hi, set(qs))
                        for q in qs:
                            _draw_box(segs_top[q], segs_mid[q], segs_bot[q], label)

            # 拼接这一层
            for q in range(self._n_qubits):
                rows[r_top(q)] += "".join(segs_top[q])
                rows[r_mid(q)] += "".join(segs_mid[q])
                rows[r_bot(q)] += "".join(segs_bot[q])

        # 处理折叠（参考 Qiskit 的分页方案）
        n_layers = len(self.layers)
        layer_width = cell_w  # 每层的字符宽度
        
        # 分页箭头字符
        arrow_right = "»" if not ascii_fallback else ">>"
        arrow_left = "«" if not ascii_fallback else "<<"
        
        # 确定每行最大字符数
        if line_length is None:
            # 自动检测终端宽度
            try:
                terminal_width, _ = get_terminal_size()
                line_length = terminal_width
            except Exception:
                line_length = 80  # 默认值
        
        # 如果 fold 为 -1，禁用折叠
        if fold == -1:
            return "\n".join(rows)
        
        # 计算每行可以显示的层数
        if fold is not None:
            layers_per_line = fold
        else:
            # 根据 line_length 自动计算
            available_width = line_length - prefix_w - 4  # 留出箭头空间
            layers_per_line = max(1, available_width // layer_width)
        
        if n_layers <= layers_per_line:
            # 不需要折叠
            return "\n".join(rows)
        
        # 需要折叠：将电路分成多段
        result_parts = []
        
        for segment_idx, segment_start in enumerate(range(0, n_layers, layers_per_line)):
            segment_end = min(segment_start + layers_per_line, n_layers)
            is_first_segment = segment_idx == 0
            is_last_segment = segment_end >= n_layers
            
            # 计算这一段的字符范围
            char_start = prefix_w + segment_start * layer_width
            char_end = prefix_w + segment_end * layer_width
            
            segment_rows = []
            for q in range(self._n_qubits):
                # 每个量子比特3行
                top_row = rows[r_top(q)]
                mid_row = rows[r_mid(q)]
                bot_row = rows[r_bot(q)]
                
                # 截取这一段
                if is_first_segment:
                    # 第一段包含前缀
                    top_seg = top_row[:char_end]
                    mid_seg = mid_row[:char_end]
                    bot_seg = bot_row[:char_end]
                else:
                    # 后续段需要添加前缀和左箭头
                    top_prefix = " " * prefix_w
                    mid_prefix = label_mid[q] + " " * (prefix_w - len(label_mid[q]))
                    bot_prefix = " " * prefix_w
                    top_seg = top_prefix + " " + top_row[char_start:char_end]
                    mid_seg = mid_prefix + arrow_left + mid_row[char_start:char_end]
                    bot_seg = bot_prefix + " " + bot_row[char_start:char_end]
                
                # 添加右箭头（如果不是最后一段）
                if not is_last_segment:
                    top_seg += " "
                    mid_seg += arrow_right
                    bot_seg += " "
                
                segment_rows.append(top_seg)
                segment_rows.append(mid_seg)
                segment_rows.append(bot_seg)
            
            result_parts.append("\n".join(segment_rows))
        
        return "\n\n".join(result_parts)
    
    def _draw_mpl(self, figsize: tuple = None):
        """使用 matplotlib 绘制电路图"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            from matplotlib.patches import FancyBboxPatch, Circle
        except ImportError:
            raise ImportError("matplotlib is required for PNG output. Install with: pip install matplotlib")
        
        # 计算图像大小
        n_layers = len(self.layers)
        if figsize is None:
            width = max(6, n_layers * 1.5 + 2)
            height = max(3, self._n_qubits * 0.8 + 1)
            figsize = (width, height)
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_xlim(-0.5, n_layers + 0.5)
        ax.set_ylim(-0.5, self._n_qubits - 0.5)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.axis('off')
        
        # 绘制量子比特线和标签
        for q in range(self._n_qubits):
            ax.hlines(q, -0.3, n_layers + 0.3, colors='black', linewidth=1)
            ax.text(-0.5, q, f'q{q}', ha='right', va='center', fontsize=10, fontweight='bold')
        
        # 门的样式
        gate_color = '#E8F4FD'
        ctrl_color = 'black'
        box_width = 0.6
        box_height = 0.5
        
        def format_param(p):
            """格式化参数显示"""
            import math
            if isinstance(p, (int, float)):
                pi_mult = p / math.pi
                if abs(pi_mult - round(pi_mult)) < 0.01 and abs(round(pi_mult)) <= 4:
                    mult = round(pi_mult)
                    if mult == 0: return "0"
                    elif mult == 1: return "π"
                    elif mult == -1: return "-π"
                    else: return f"{mult}π"
                else:
                    return f"{p:.2f}"
            return str(p)[:4]
        
        def draw_gate_box(x, y, label, color=gate_color):
            """绘制门的方框"""
            rect = FancyBboxPatch(
                (x - box_width/2, y - box_height/2),
                box_width, box_height,
                boxstyle="round,pad=0.02,rounding_size=0.1",
                facecolor=color, edgecolor='black', linewidth=1.5
            )
            ax.add_patch(rect)
            fontsize = 9 if len(label) <= 4 else 7 if len(label) <= 6 else 5
            ax.text(x, y, label, ha='center', va='center', fontsize=fontsize, fontweight='bold')
        
        def draw_control(x, y):
            """绘制控制点"""
            circle = Circle((x, y), 0.08, facecolor=ctrl_color, edgecolor=ctrl_color)
            ax.add_patch(circle)
        
        def draw_target_x(x, y):
            """绘制 CNOT 目标 (⊕)"""
            circle = Circle((x, y), 0.2, facecolor='white', edgecolor='black', linewidth=1.5)
            ax.add_patch(circle)
            ax.plot([x - 0.2, x + 0.2], [y, y], 'k-', linewidth=1.5)
            ax.plot([x, x], [y - 0.2, y + 0.2], 'k-', linewidth=1.5)
        
        def draw_swap(x, y):
            """绘制 SWAP 符号 (×)"""
            size = 0.12
            ax.plot([x - size, x + size], [y - size, y + size], 'k-', linewidth=2)
            ax.plot([x - size, x + size], [y + size, y - size], 'k-', linewidth=2)
        
        # 绘制每一层的门
        for layer_idx, layer in enumerate(self.layers):
            x = layer_idx + 0.5
            
            for inst in layer:
                qs = list(inst.qubits)
                name = inst.name
                params = inst.params if hasattr(inst, 'params') else []
                
                if not qs:
                    continue
                
                # 构建标签
                if params:
                    param_str = ",".join(format_param(p) for p in params[:2])
                    label = f"{name}({param_str})"
                else:
                    label = name.upper() if len(name) <= 2 else name
                
                if name == "cx":
                    c, t = qs[0], qs[1]
                    ax.vlines(x, min(c, t), max(c, t), colors='black', linewidth=1.5)
                    draw_control(x, c)
                    draw_target_x(x, t)
                
                elif name in ("cz", "cp"):
                    c, t = qs[0], qs[1]
                    ax.vlines(x, min(c, t), max(c, t), colors='black', linewidth=1.5)
                    draw_control(x, c)
                    draw_control(x, t)
                
                elif name in ("crz", "crx", "cry", "cu", "cu1", "cu3", "ch", "cy"):
                    c, t = qs[0], qs[1]
                    ax.vlines(x, min(c, t), max(c, t), colors='black', linewidth=1.5)
                    draw_control(x, c)
                    base_name = name[1:].upper() if name.startswith('c') else name.upper()
                    if params:
                        param_str = ",".join(format_param(p) for p in params[:1])
                        target_label = f"{base_name}({param_str})"
                    else:
                        target_label = base_name
                    draw_gate_box(x, t, target_label)
                
                elif name in ("ccx", "c3x", "c4x"):
                    controls, target = qs[:-1], qs[-1]
                    ax.vlines(x, min(qs), max(qs), colors='black', linewidth=1.5)
                    for c in controls:
                        draw_control(x, c)
                    draw_target_x(x, target)
                
                elif name == "ccz":
                    ax.vlines(x, min(qs), max(qs), colors='black', linewidth=1.5)
                    for q in qs:
                        draw_control(x, q)
                
                elif name == "swap":
                    a, b = qs[0], qs[1]
                    ax.vlines(x, min(a, b), max(a, b), colors='black', linewidth=1.5)
                    draw_swap(x, a)
                    draw_swap(x, b)
                
                elif name == "cswap":
                    c, a, b = qs[0], qs[1], qs[2]
                    ax.vlines(x, min(qs), max(qs), colors='black', linewidth=1.5)
                    draw_control(x, c)
                    draw_swap(x, a)
                    draw_swap(x, b)
                
                elif name == "barrier":
                    for q in qs:
                        ax.axvline(x, color='gray', linestyle='--', linewidth=1, alpha=0.5)
                
                elif name == "measure":
                    q = qs[0]
                    draw_gate_box(x, q, "M", color='#FFE4B5')
                
                else:
                    # 单比特门或其他门
                    if len(qs) == 1:
                        draw_gate_box(x, qs[0], label)
                    else:
                        ax.vlines(x, min(qs), max(qs), colors='black', linewidth=1.5)
                        for q in qs:
                            draw_gate_box(x, q, label)
        
        plt.tight_layout()
        return fig
    
    def __len__(self) -> int:
        """返回层数"""
        return self.depth
    
    def __iter__(self) -> Iterator[Layer]:
        """迭代层"""
        return iter(self.layers)
    
    def __getitem__(self, index: int) -> Layer:
        """获取指定层"""
        return self.layers[index]
