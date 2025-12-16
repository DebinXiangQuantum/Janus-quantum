"""
Janus 量子门基类

定义量子门的基本结构和接口
"""
from typing import List, Optional, Union
import numpy as np
from .operation import Operation


class Gate(Operation):
    """
    量子门基类
    
    量子门是酉操作，可以用酉矩阵表示
    
    Attributes:
        name: 门的名称
        qubits: 门作用的量子比特索引列表
        params: 门的参数列表（如旋转角度）
        label: 可选的标签
    """
    
    def __init__(
        self,
        name: str,
        num_qubits: int,
        params: Optional[List[float]] = None,
        label: Optional[str] = None
    ):
        self._name = name
        self._num_qubits = num_qubits
        self._params = params if params is not None else []
        self._label = label
        self._qubits: List[int] = []  # 实际作用的量子比特，在添加到电路时设置
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def num_qubits(self) -> int:
        return self._num_qubits
    
    @property
    def params(self) -> List[float]:
        return self._params
    
    @params.setter
    def params(self, value: List[float]):
        self._params = value
    
    @property
    def qubits(self) -> List[int]:
        return self._qubits
    
    @qubits.setter
    def qubits(self, value: List[int]):
        if len(value) != self._num_qubits:
            raise ValueError(f"Gate {self._name} requires {self._num_qubits} qubits, got {len(value)}")
        self._qubits = value
    
    @property
    def label(self) -> Optional[str]:
        return self._label
    
    @label.setter
    def label(self, value: str):
        self._label = value
    
    def to_matrix(self) -> np.ndarray:
        """
        返回门的酉矩阵表示
        
        子类应该重写此方法
        """
        raise NotImplementedError(f"to_matrix not implemented for {self._name}")
    
    def inverse(self) -> 'Gate':
        """
        返回门的逆操作
        
        子类应该重写此方法
        """
        raise NotImplementedError(f"inverse not implemented for {self._name}")
    
    def copy(self) -> 'Gate':
        """创建门的副本"""
        new_gate = Gate(self._name, self._num_qubits, self._params.copy(), self._label)
        new_gate._qubits = self._qubits.copy()
        return new_gate
    
    def __repr__(self) -> str:
        if self._params:
            return f"{self._name}({', '.join(map(str, self._params))})"
        return self._name
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, Gate):
            return False
        return (self._name == other._name and 
                self._num_qubits == other._num_qubits and
                self._params == other._params)
    
    def to_dict(self) -> dict:
        """转换为字典格式（兼容旧格式）"""
        return {
            'name': self._name,
            'qubits': self._qubits,
            'params': self._params
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Gate':
        """从字典创建门"""
        gate = cls(data['name'], len(data['qubits']), data.get('params', []))
        gate.qubits = data['qubits']
        return gate
