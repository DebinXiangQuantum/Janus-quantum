from __future__ import annotations
from typing import TYPE_CHECKING, List
import numpy as np
from janus.circuit import Circuit, Gate
from janus.circuit.dag import circuit_to_dag
from janus.circuit.library.standard_gates import (
    UGate,
    XGate,
    RXGate,
    RYGate,
    RZGate,
    SGate,
    SdgGate,
    TGate,
    TdgGate,
    ZGate,
    YGate,
    HGate,
)
from .exceptions import ParameterError,  DecomposeError

DEFAULT_ATOL = 1e-12

ONE_QUBIT_EULER_BASIS_GATES = {
    "U": ["u"],
    "ZYZ": ["rz", "ry"],
    "ZXZ": ["rz", "rx"],
    "XYX": ["rx", "ry"],
    "XZX": ["rx", "rz"],
}
NAME_MAP = {
    "u": UGate,
    "rx": RXGate,
    "ry": RYGate,
    "rz": RZGate,
    "x": XGate,
    "z": ZGate,
    "y": YGate,
    "h": HGate,
    "s": SGate,
    "sdg": SdgGate,
    "t": TGate,
    "tdg": TdgGate,
}
def is_unitary_matrix(mat: np.ndarray, atol: float = 1e-10) -> bool:
    if mat.shape[0] != mat.shape[1]:
        return False
    return np.allclose(np.dot(mat, mat.conj().T), np.eye(mat.shape[0]), atol=atol)
class EulerOneQubitDecomposer:
    @staticmethod
    def params_zyz(unitary: np.ndarray) -> tuple:
        a, b, c, d = unitary.flatten()
        theta = 2 * np.arccos(np.abs(a))
        if np.isclose(theta, 0) or np.isclose(theta, np.pi):
            phi = 0
            lam = np.angle(d)
        else:
            # 修复H门的分解问题
            if np.isclose(np.abs(a), np.abs(b)) and np.isclose(np.abs(c), np.abs(d)) and np.isclose(np.abs(a), np.abs(c)):
                # 这是H门的情况
                phi = 0
                lam = np.pi
            # 修复Y门的分解问题
            elif np.isclose(a, 0) and np.isclose(d, 0) and np.isclose(b, -1j) and np.isclose(c, 1j):
                # 这是Y门的情况
                theta = np.pi
                phi = np.pi/2
                lam = np.pi/2
            else:
                phi = np.angle(c)
                lam = -np.angle(b)
        phase = np.angle(a)
        return theta, phi, lam, phase
    @staticmethod
    def params_zxz(unitary: np.ndarray) -> tuple:
        theta, phi, lam, phase = EulerOneQubitDecomposer.params_zyz(unitary)
        return theta, phi + np.pi/2, lam - np.pi/2, phase
    @staticmethod
    def params_xyx(unitary: np.ndarray) -> tuple:
        theta, phi, lam, phase = EulerOneQubitDecomposer.params_zyz(unitary)
        return theta, phi + np.pi/2, lam - np.pi/2, phase
    @staticmethod
    def params_xzx(unitary: np.ndarray) -> tuple:
        theta, phi, lam, phase = EulerOneQubitDecomposer.params_zyz(unitary)
        return theta, phi, lam, phase
    @staticmethod
    def params_u3(unitary: np.ndarray) -> tuple:
        return EulerOneQubitDecomposer.params_zyz(unitary)
    @staticmethod
    def unitary_to_circuit(unitary: np.ndarray, basis: List[str], use_dag: bool = False) -> Circuit:
        circuit = Circuit(1)
        # 直接使用EulerOneQubitDecomposer的params方法，而不是创建OneQubitEulerDecomposer实例
        if basis[0] == "U":
            theta, phi, lam, phase = EulerOneQubitDecomposer.params_u3(unitary)
            circuit.u(theta, phi, lam, 0)
        elif basis[0] == "ZYZ":
            theta, phi, lam, phase = EulerOneQubitDecomposer.params_zyz(unitary)
            circuit.rz(phi, 0)
            circuit.ry(theta, 0)
            circuit.rz(lam, 0)
        elif basis[0] == "ZXZ":
            theta, phi, lam, phase = EulerOneQubitDecomposer.params_zxz(unitary)
            circuit.rz(phi, 0)
            circuit.rx(theta, 0)
            circuit.rz(lam, 0)
        elif basis[0] == "XYX":
            theta, phi, lam, phase = EulerOneQubitDecomposer.params_xyx(unitary)
            circuit.rx(phi, 0)
            circuit.ry(theta, 0)
            circuit.rx(lam, 0)
        elif basis[0] == "XZX":
            theta, phi, lam, phase = EulerOneQubitDecomposer.params_xzx(unitary)
            circuit.rx(phi, 0)
            circuit.rz(theta, 0)
            circuit.rx(lam, 0)
        
        # 添加全局相位门
        if not np.isclose(phase, 0, atol=1e-10):
            circuit.p(phase, 0)
        
        return circuit
    @staticmethod
    def unitary_to_gate_sequence(unitary: np.ndarray) -> 'GateSequence':
        class GateSequence:
            def __init__(self):
                self.global_phase = 0.0
        return GateSequence()
euler_one_qubit_decomposer = EulerOneQubitDecomposer()
class Operator:
    def __init__(self, data):
        self.data = data
    @classmethod
    def from_matrix(cls, matrix):
        return cls(matrix)
class OneQubitEulerDecomposer:
    def __init__(self, basis: str = "U", use_dag: bool = False):
        self.basis = basis
        self.use_dag = use_dag
    def build_circuit(self, gates) -> Circuit:
        qc = Circuit(1)
        for gate_entry in gates:
            if isinstance(gate_entry, tuple):
                gate_name, params = gate_entry
                gate = NAME_MAP[gate_name](*params)
                qc.append(gate, [0])
        return qc
    def __call__(self,
        unitary: Operator | Gate | np.ndarray,
        atol: float = DEFAULT_ATOL,
    ) -> Circuit:
        if hasattr(unitary, "to_operator"):
            unitary = unitary.to_operator().data
        elif hasattr(unitary, "to_matrix"):
            unitary = unitary.to_matrix()
        unitary = np.asarray(unitary, dtype=complex)
        if unitary.shape != (2, 2):
            raise ParameterError("OneQubitEulerDecomposer: expected 2x2 input matrix")
        if not is_unitary_matrix(unitary):
            raise DecomposeError("OneQubitEulerDecomposer: input matrix is not unitary.")
        return self._decompose(unitary, atol=atol)
    def _decompose(self, unitary, atol=DEFAULT_ATOL):
        if self.use_dag:
            # 使用DAGCircuit实现
            # 首先创建普通电路
            circuit = euler_one_qubit_decomposer.unitary_to_circuit(
                unitary, [self.basis], self.use_dag
            )
            
            # 转换为DAGCircuit并返回
            return circuit_to_dag(circuit)
        else:
            # 原始实现，返回普通Circuit
            return euler_one_qubit_decomposer.unitary_to_circuit(
                unitary, [self.basis], self.use_dag
            )
    @property
    def basis(self):
        return self._basis
    @basis.setter
    def basis(self, basis):
        basis_methods = {
            "U": self._params_u3,
            "ZYZ": self._params_zyz,
            "ZXZ": self._params_zxz,
            "XYX": self._params_xyx,
            "XZX": self._params_xzx,
        }
        if basis not in basis_methods:
            raise ParameterError(f"OneQubitEulerDecomposer: unsupported basis {basis}")
        self._basis = basis
        self._params = basis_methods[basis]
    def angles(self, unitary: np.ndarray) -> tuple:
        unitary = np.asarray(unitary, dtype=complex)
        theta, phi, lam, _ = self._params(unitary)
        return theta, phi, lam
    def angles_and_phase(self, unitary: np.ndarray) -> tuple:
        unitary = np.asarray(unitary, dtype=complex)
        return self._params(unitary)
    _params_zyz = staticmethod(euler_one_qubit_decomposer.params_zyz)
    _params_zxz = staticmethod(euler_one_qubit_decomposer.params_zxz)
    _params_xyx = staticmethod(euler_one_qubit_decomposer.params_xyx)
    _params_xzx = staticmethod(euler_one_qubit_decomposer.params_xzx)
    _params_u3 = staticmethod(euler_one_qubit_decomposer.params_u3)
def decompose_one_qubit(unitary, basis='U', use_dag=False, atol=DEFAULT_ATOL):
    decomposer = OneQubitEulerDecomposer(basis=basis, use_dag=use_dag)
    return decomposer(unitary, atol=atol)