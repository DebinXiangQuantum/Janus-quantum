'''
Author: name/jxhhhh 2071379252@qq.com
Date: 2024-04-17 06:05:07
LastEditors: name/jxhhhh 2071379252@qq.com
LastEditTime: 2024-04-19 03:06:03
FilePath: /JanusQ/janusq/data_objects/circuit.py
Description: 
    量子电路数据结构与转换模块
    
    本模块定义了 JanusQ 框架中量子电路的核心数据结构，包括：
    - Gate: 量子门的基本表示
    - Layer: 电路层（可并行执行的门集合）
    - Circuit: 完整的量子电路
    - SeperatableCircuit: 可分离电路（多个独立电路的组合）
    
    同时提供了与 Qiskit QuantumCircuit 之间的双向转换功能。

Copyright (c) 2024 by name/jxhhhh 2071379252@qq.com, All Rights Reserved. 
'''
import copy
import logging
import uuid
from qiskit import QuantumCircuit
from qiskit.circuit import Instruction
from qiskit.dagcircuit import DAGCircuit, DAGOpNode
from functools import lru_cache, reduce
from copy import deepcopy
from typing import List
import numpy as np
class Gate(dict):
    """
    量子门类，继承自 dict
    
    用于表示单个量子门，包含门的名称、作用的量子比特和参数信息。
    
    Attributes:
        layer_index (int): 该门所在的层索引
        index (int): 该门在整个电路中的全局索引
        vec: 门的向量表示（用于特定算法）
    
    Example:
        gate = Gate({
            'name': 'rx',
            'qubits': [0],
            'params': [3.14]
        })
    """
    def __init__(self, gate: dict, layer_index: int = None, copy = True):
        """
        初始化量子门
        
        Args:
            gate (dict): 门的字典表示，必须包含 'name' 和 'qubits' 键
            layer_index (int, optional): 门所在的层索引
            copy (bool): 是否深拷贝输入的 gate 字典，默认为 True
        """
        assert 'qubits' in gate
        assert 'name' in gate
        if copy:
            gate = deepcopy(gate)
        self.layer_index = layer_index
        self.index: int = None
        self.vec = None
        super().__init__(gate)

    @property
    def qubits(self):
        """获取门作用的量子比特列表"""
        return self['qubits']

    @property
    def name(self):
        """获取门的名称（如 'rx', 'cx' 等）"""
        return self['name']

    @property
    def params(self):
        """获取门的参数列表（如旋转角度）"""
        return self['params']


class Layer(list):
    """
    电路层类，继承自 list
    
    表示量子电路中的一层，包含可以并行执行的量子门集合。
    同一层中的门作用在不同的量子比特上，因此可以同时执行。
    """
    def __init__(self, gates: List[Gate], layer_index: int = None, copy=True):
        """
        初始化电路层
        
        Args:
            gates (List[Gate]): 该层包含的量子门列表
            layer_index (int, optional): 层的索引
            copy (bool): 是否深拷贝门对象，默认为 True
        """
        if copy:
            gates = [
                Gate(gate, layer_index, copy = copy)
                for gate in gates
            ]
        super().__init__(gates)


class Circuit(list):
    '''
    量子电路类，继承自 list
    
    表示完整的量子电路，由多个电路层（Layer）组成。
    每一层包含可以并行执行的量子门。
    
    TODO: 电路应该是不可变的（常量）
    '''

    def __init__(self, layers: List[Layer], n_qubits = None, copy=True, measured_qubits = None, operated_qubits = None):
        """
        初始化量子电路
        
        Args:
            layers (List[Layer]): 电路的层列表，或 QuantumCircuit 对象，或另一个 Circuit 对象
            n_qubits (int, optional): 电路中量子比特的总数。如果为 None，将从门的作用比特推导
            copy (bool): 是否深拷贝输入的层，默认为 True
            measured_qubits (list[int], optional): 需要测量的量子比特列表
            operated_qubits (list[int], optional): 实际操作的量子比特列表。如果为 None，将自动计算
        
        Raises:
            AssertionError: 如果 gate 字典缺少必要的键
        """
        # 为电路分配唯一的 UUID 标识符
        # TODO: 使用 id 进行缓存优化
        self.id = uuid.uuid1()
        
        # 如果输入是另一个 Circuit 对象且未指定 n_qubits，则继承其量子比特数
        if isinstance(layers, Circuit) and n_qubits is None:
            n_qubits = layers.n_qubits
        
        # 如果输入是 Qiskit QuantumCircuit，则转换为本框架的 Circuit 格式
        if isinstance(layers, QuantumCircuit):
            if n_qubits is None:
                n_qubits = layers.num_qubits
            layers = qiskit_to_circuit(layers)

        # 如果需要深拷贝，则为每一层创建新的 Layer 对象
        if copy:
            layers = [
                Layer(layer, index, copy)
                for index, layer in enumerate(layers)
            ]

        # 调用父类 list 的初始化方法
        super().__init__(layers)

        # 将所有层中的门展平为一维列表，并按顺序排列
        self.gates: list[Gate] = self._sort_gates()

        # 如果未指定量子比特数，则从所有门的作用比特中推导
        # 取最大的比特索引 + 1 作为总量子比特数
        if n_qubits is None:
            n_qubits = max(reduce(lambda a, b: a + b,
                           [gate['qubits'] for gate in self.gates]))
        self.n_qubits = n_qubits

        # 确定实际操作的量子比特集合
        # 如果未指定，则自动计算所有被门作用的量子比特
        if operated_qubits is None:
            self.operated_qubits = self._sort_qubits()  # 实际操作的量子比特
        else:
            self.operated_qubits = operated_qubits

        # 为所有门分配全局索引
        self._assign_index()
        
        # 电路的名称（可选）
        self.name : str = None
        
        # 需要测量的量子比特列表
        self.measured_qubits: list[int] = measured_qubits
    

    def rx(self, angle, qubit, layer_index):
        """
        在指定层添加 RX 旋转门（绕 X 轴旋转）
        
        Args:
            angle (float): 旋转角度（弧度）
            qubit (int): 作用的量子比特索引
            layer_index (int): 目标层的索引
        """
        if layer_index >= len(self):
            # 如果层索引超出范围，创建新层
            self.append(Layer([Gate({'name': 'rx', 'qubits': [qubit], 'params': [angle]})]))
        else:
            # 否则在现有层中添加门
            self[layer_index].append(Gate({'name': 'rx', 'qubits': [qubit], 'params': [angle]}))

    def crz(self, angle, qubit1, qubit2, layer_index):
        """
        在指定层添加 CRZ 受控旋转门（受控 Z 轴旋转）
        
        Args:
            angle (float): 旋转角度（弧度）
            qubit1 (int): 控制比特索引
            qubit2 (int): 目标比特索引
            layer_index (int): 目标层的索引
        """
        if layer_index >= len(self):
            # 如果层索引超出范围，创建新层
            self.append(Layer([Gate({'name': 'crz', 'qubits': [qubit1, qubit2], 'params': [angle]})]))
        else:
            # 否则在现有层中添加门
            self[layer_index].append(Gate({'name': 'crz', 'qubits': [qubit1, qubit2], 'params': [angle]}))

    def ry(self, angle, qubit, layer_index):
        """
        在指定层添加 RY 旋转门（绕 Y 轴旋转）
        
        Args:
            angle (float): 旋转角度（弧度）
            qubit (int): 作用的量子比特索引
            layer_index (int): 目标层的索引
        """
        if layer_index >= len(self):
            # 如果层索引超出范围，创建新层
            self.append(Layer([Gate({'name': 'ry', 'qubits': [qubit], 'params': [angle]})]))
        else:
            # 否则在现有层中添加门
            self[layer_index].append(Gate({'name': 'ry', 'qubits': [qubit], 'params': [angle]}))

    def x(self, qubit, layer_index):
        """
        在指定层添加 X 门（Pauli X 门，相当于经典的 NOT 门）
        
        Args:
            qubit (int): 作用的量子比特索引
            layer_index (int): 目标层的索引
        """
        if layer_index >= len(self):
            # 如果层索引超出范围，创建新层
            self.append(Layer([Gate({'name': 'x', 'qubits': [qubit], 'params': []})]))
        else:
            # 否则在现有层中添加门
            self[layer_index].append(Gate({'name': 'x', 'qubits': [qubit], 'params': []}))

    @property
    def num_two_qubit_gate(self):
        """
        计算电路中两比特门的数量
        
        Returns:
            int: 两比特门的总数（如 CX、CZ、CRZ 等）
        """
        cnt = 0
        for gate in self.gates:
            if len(gate.qubits) > 1:
                cnt += 1
        return cnt
    
    @property
    def duration(self, single_qubit_gate_duration=30, two_qubit_gate_duration=60):
        """
        估算电路的执行时间
        
        基于每层中最复杂的门类型来计算总时间。
        单比特门执行时间：30 时间单位
        两比特门执行时间：60 时间单位
        
        Args:
            single_qubit_gate_duration (int): 单比特门的执行时间，默认 30
            two_qubit_gate_duration (int): 两比特门的执行时间，默认 60
        
        Returns:
            int: 总执行时间
        """
        # 获取每层中最复杂的门（比特数最多的门）
        layer_types = [max([len(gate.qubits) for gate in layer]) for layer in self]
        duration = 0
        for layer_type in layer_types:
            if layer_type == 1:
                # 该层只有单比特门
                duration += single_qubit_gate_duration
            elif layer_type == 2:
                # 该层有两比特门
                duration += two_qubit_gate_duration
        return duration
    
    @property
    def depth(self):
        """
        获取电路的深度（层数）
        
        Returns:
            int: 电路中的层数
        """
        return len(self)
    
    @property
    def n_gates(self):
        """
        获取电路中的总门数
        
        Returns:
            int: 电路中所有门的总数
        """
        return len(self.gates)

    def to_qiskit(self, barrier=True) -> QuantumCircuit:
        """
        将本框架的 Circuit 转换为 Qiskit QuantumCircuit
        
        Args:
            barrier (bool): 是否在每层之间添加 barrier（栅栏），默认为 True
        
        Returns:
            QuantumCircuit: 转换后的 Qiskit 量子电路
        """
        return circuit_to_qiskit(self, barrier=barrier)

    def __str__(self) -> str:
        """
        获取电路的字符串表示
        
        Returns:
            str: 转换为 Qiskit 格式后的字符串表示
        """
        return str(self.to_qiskit())

    def __add__(self, other: List[Layer]):
        """
        电路连接操作符（+）
        
        将两个电路连接在一起，形成新的电路。
        量子比特数取两个电路中的最大值。
        
        Args:
            other (List[Layer] | Circuit): 要连接的电路或层列表
        
        Returns:
            Circuit: 新的连接后的电路
        """
        if isinstance(other, Circuit):
            # 取两个电路中量子比特数的最大值
            n_qubits = max([self.n_qubits, other.n_qubits])
        else:
            n_qubits = self.n_qubits
        return Circuit(list.__add__(self.copy(), other.copy()), n_qubits)

    def copy(self):
        """
        创建电路的深拷贝
        
        Returns:
            Circuit: 电路的完整副本
        """
        return copy.deepcopy(self)



    def _sort_gates(self,):
        """
        将所有层中的门展平为一维列表
        
        遍历所有层，将其中的门按顺序提取出来，形成一个一维的门列表。
        这样便于按全局顺序访问所有门。
        
        Returns:
            list[Gate]: 展平后的门列表
        """
        self.gates: list[Gate] = [
            gate
            for layer in self
            for gate in layer
        ]
        self._assign_index()
        return self.gates

    def _sort_qubits(self,):
        """
        提取电路中实际操作的所有量子比特
        
        遍历所有门，收集它们作用的量子比特，去重后返回。
        
        Returns:
            list[int]: 实际操作的量子比特列表（已去重）
        """
        operated_qubits = []
        for gate in self.gates:
            operated_qubits += gate.qubits
        self.operated_qubits = list(set(operated_qubits))  # 实际操作的量子比特
        return self.operated_qubits

    def _assign_index(self):
        """
        为所有门分配全局索引
        
        遍历 self.gates 列表，为每个门设置其全局索引（在整个电路中的位置）。
        """
        for index, gate in enumerate(self.gates):
            gate.index = index
            
    def _assign_layer_index(self):
        """
        为所有门分配层索引
        
        遍历所有层，为每层中的门设置其所在层的索引。
        """
        for layer_index, layer in enumerate(self):
            for gate in layer:
                gate.layer_index = layer_index   
                
    def clean_empty_layer(self):
        """
        清理电路中的空层
        
        移除所有空的层（不包含任何门的层），并重新分配层索引。
        """
        while [] in self:
            self.remove([])
        self._assign_layer_index()
            
    def get_available_space(self, target_gate: Gate):
        """
        获取目标门可以移动的层范围
        
        根据量子比特的依赖关系，找出目标门可以移动到的层范围。
        门只能移动到不与其他门冲突的层（即不作用在相同的量子比特上）。
        
        Args:
            target_gate (Gate): 目标门
        
        Returns:
            range: 可用的层索引范围 [former_layer_index, next_layer_index)
        
        Raises:
            AssertionError: 如果目标门不在电路中
        """
        assert target_gate in self.gates
        gate_qubits = target_gate.qubits

        # 向前查找：找到最后一个与目标门冲突的层
        if target_gate.layer_index != 0:
            former_layer_index = target_gate.layer_index - 1
            while True:
                now_layer = self[former_layer_index]
                # 获取该层中所有门作用的量子比特
                layer_qubits = reduce(lambda a, b: a+b, [gate['qubits'] for gate in now_layer])
                # 如果该层与目标门冲突（共享量子比特）或已到达第一层，则停止
                if any([qubit in layer_qubits for qubit in gate_qubits]) or former_layer_index == 0:
                    break
                former_layer_index -= 1
        else:
            former_layer_index = target_gate.layer_index

        # 向后查找：找到第一个与目标门冲突的层
        if target_gate.layer_index != len(self)-1:
            next_layer_index = target_gate.layer_index + 1
            while True:
                now_layer = self[next_layer_index]
                # 获取该层中所有门作用的量子比特
                layer_qubits = reduce(lambda a, b: a+b, [gate['qubits'] for gate in now_layer])
                # 如果该层与目标门冲突（共享量子比特）或已到达最后一层，则停止
                if any([qubit in layer_qubits for qubit in gate_qubits]) or next_layer_index == len(self)-1:
                    break
                next_layer_index += 1
        else:
            next_layer_index = target_gate.layer_index

        # 返回可用的层范围（不包括冲突的层）
        return range(former_layer_index, next_layer_index)
    
   
    def move(self, gate: Gate, new_layer: int):
        """
        将指定的门移动到新的层
        
        创建电路的副本，将指定的门从原层移除，添加到新层，
        然后更新所有索引和清理空层。
        
        Args:
            gate (Gate): 要移动的门
            new_layer (int): 目标层的索引
        
        Returns:
            Circuit: 移动后的新电路（原电路不变）
        
        Raises:
            AssertionError: 如果门不在电路中
        """
        assert gate in self.gates
  
        # 创建电路的深拷贝，以保持原电路不变
        new_circuit = self.copy()
        
        # 在新电路中找到对应的门
        new_gate: Gate = new_circuit.gates[gate.index]
        
        # 获取门当前所在的层
        now_layer: Layer = new_circuit[new_gate.layer_index]
        
        # 从原层中移除该门
        now_layer.remove(new_gate)
        
        # 将门添加到新层
        new_circuit[new_layer].append(new_gate)
        
        # 更新门的层索引
        new_gate.layer_index = new_layer
        
        # 重新排序和索引所有门
        new_circuit._sort_gates()
        new_circuit._assign_index()
        
        # 清理空层
        new_circuit.clean_empty_layer()
        
        return new_circuit
        


class SeperatableCircuit(Circuit):
    """
    可分离电路类，继承自 Circuit
    
    表示由多个独立的子电路组合而成的电路。
    这些子电路作用在不同的量子比特集合上，可以并行执行。
    
    Example:
        circuit1 = Circuit([...])  # 作用在比特 0, 1
        circuit2 = Circuit([...])  # 作用在比特 2, 3
        sep_circuit = SeperatableCircuit([circuit1, circuit2], n_qubits=4)
    """
    
    def __init__(self, seperatable_circuits: List[Circuit], n_qubits):
        """
        初始化可分离电路
        
        将多个独立的子电路按层对齐，形成一个整体电路。
        每一层包含所有子电路在该层的门。
        
        Args:
            seperatable_circuits (List[Circuit]): 子电路列表
            n_qubits (int): 总量子比特数
        """
        # 找到所有子电路中最多的层数
        max_layer = max([len(c) for c in seperatable_circuits])

        # 构建整体电路
        overall_circuit = []
        for layer_index in range(max_layer):
            # 创建新的层
            overall_layer = Layer([])
            
            # 遍历所有子电路
            for c in seperatable_circuits:
                # 如果该子电路在该层有内容，则将其门添加到整体层
                if len(c) > layer_index:
                    for gate in c[layer_index]:
                        overall_layer.append(gate)

            overall_circuit.append(overall_layer)

        # 保存原始的子电路列表
        self.seperatable_circuits = seperatable_circuits

        # 调用父类初始化
        super().__init__(overall_circuit, n_qubits, copy=False)


'''
    Circuit of JanusQ is represented of a 2-d list, named layer_circuit:
    [
        [{
            'name': 'CX',
            'qubits': [0, 1],
            'params': [],
        },{
            'name': 'RX',
            'qubits': [0],
            'params': [np.pi/2],
        }],
        [{
            'name': 'CX',
            'qubits': [2, 3],
            'params': [],
        }]
    ]
'''
import math

def handle_params(params):
    """
    处理门的参数，将其转换为标准格式
    
    支持两种输入格式：
    1. 字典格式：{'param_name': value}
    2. 列表/元组格式：[value1, value2, ...]
    
    对于字符串参数（如 'pi/2'），将其转换为对应的数值。
    
    Args:
        params (dict | list | tuple): 门的参数
    
    Returns:
        list: 转换后的参数列表
    
    Example:
        handle_params({'angle': 'pi/2'}) -> [1.5707963...]
        handle_params([0, 'pi/4']) -> [0, 0.7853981...]
    """
    # 如果参数是字典格式
    if isinstance(params, dict):
        if params == {}:
            # 空字典返回空列表
            return []
        ps = list(params.values())
    else:
        # 否则直接使用参数列表
        ps = params
    
    result = []
    for p in ps:
        if p == 0:
            # 0 保持不变
            result.append(p)
        else:
            # 对于字符串参数（如 'pi/2'），提取分母并计算 π/分母
            # 例如 'pi/2' -> 提取 '2' -> π/2
            result.append(math.pi / float(p[3:]))
    return result

# def circuit_to_qiskit(circuit: Circuit, n_qubits=None, barrier=True) -> QuantumCircuit:
#     if n_qubits is None:
#         qiskit_circuit = QuantumCircuit(4)
#     else:
#         qiskit_circuit = QuantumCircuit(n_qubits)

#     for layer in circuit:
#         for gate in layer:
#             name = gate['name']
#             qubits = gate['qubits']
#             params = gate['params']
#             if isinstance(params, dict):
#                 params = handle_params(params)
#             if name in ('rx', 'ry', 'rz'):
#                 assert len(params) == 1 and len(qubits) == 1
#                 qiskit_circuit.__getattribute__(
#                     name)(float(params[0]), qubits[0])
#             elif name in ('cz', 'cx'):
#                 assert len(params) == 0 and len(qubits) == 2
#                 qiskit_circuit.__getattribute__(name)(qubits[0], qubits[1])
#             elif name in ('h', 'x'):
#                 qiskit_circuit.__getattribute__(name)(qubits[0])
#             elif name in ('u', 'u3', 'u1', 'u2'):
#                 '''TODO: 参数的顺序需要check下， 现在是按照pennylane的Rot的'''
#                 qiskit_circuit.__getattribute__(name)(
#                     *[float(param) for param in params], qubits[0])
#             elif name in ('crz',):
#                 qiskit_circuit.__getattribute__(name)(
#                     *[float(param) for param in params], *qubits)
#             else:
#                 gate_name = gate['name']
#                 # circuit.__getattribute__(name)(*(params + qubits))
#                 print(f'Unknown gate: {gate_name}. gate list[rx, ry, rz, cz, cx, h, x, u, u3, u1, u2, crz]')
#                 # raise Exception('unkown gate', gate)

#         if barrier:
#             qiskit_circuit.barrier()

#     return qiskit_circuit
def circuit_to_qiskit(circuit: Circuit, n_qubits=None, barrier=True) -> QuantumCircuit:
    """
    将本框架的 Circuit 转换为 Qiskit QuantumCircuit
    
    遍历电路的所有层和门，将其转换为对应的 Qiskit 门操作。
    支持的门类型：
    - 单比特旋转门：rx, ry, rz
    - 两比特门：cx, cz, crz
    - 单比特门：h, x
    - 通用单比特门：u, u1, u2, u3
    
    Args:
        circuit (Circuit): 要转换的电路
        n_qubits (int, optional): 量子比特数。如果为 None，默认为 4
        barrier (bool): 是否在每层之间添加 barrier，默认为 True
    
    Returns:
        QuantumCircuit: 转换后的 Qiskit 量子电路
    """
    # 初始化 Qiskit 电路
    if n_qubits is None:
        qiskit_circuit = QuantumCircuit(4)
    else:
        qiskit_circuit = QuantumCircuit(n_qubits)

    # 遍历电路的每一层
    for layer in circuit:
        # 遍历该层中的每个门
        for gate in layer:
            name = gate['name']
            qubits = gate['qubits']
            params = gate['params']
            
            # 处理参数格式
            if isinstance(params, dict):
                params = handle_params(params)
            
            # 将 u3 转换为 u（Qiskit 新版本的标准）
            if name == 'u3':
                name = 'u'
                
            # 处理单比特旋转门（rx, ry, rz）
            if name in ('rx', 'ry', 'rz'):
                assert len(params) == 1 and len(qubits) == 1
                qiskit_circuit.__getattribute__(
                    name)(float(params[0]), qubits[0])
            
            # 处理两比特门（cx, cz）
            elif name in ('cz', 'cx'):
                assert len(params) == 0 and len(qubits) == 2
                qiskit_circuit.__getattribute__(name)(qubits[0], qubits[1])
            
            # 处理 Pauli 门（h, x）
            elif name in ('h', 'x'):
                qiskit_circuit.__getattribute__(name)(qubits[0])
            
            # 处理通用单比特门（u, u1, u2）
            elif name in ('u', 'u1', 'u2'):
                """
                处理 u 门系列：
                - u(theta, phi, lambda)：三参数通用门
                - u2(phi, lambda)：等价于 u(π/2, phi, lambda)
                - u1(lambda)：等价于 u(0, 0, lambda)
                """
                if name == 'u1' and len(params) == 1:
                    # u1(lambda) 转换为 u(0, 0, lambda)
                    qiskit_circuit.u(0, 0, float(params[0]), qubits[0])
                elif name == 'u2' and len(params) == 2:
                    # u2(phi, lambda) 转换为 u(π/2, phi, lambda)
                    qiskit_circuit.u(np.pi/2, float(params[0]), float(params[1]), qubits[0])
                elif name == 'u' and len(params) == 3:
                    # u(theta, phi, lambda) - 三个参数
                    qiskit_circuit.u(
                        float(params[0]),  # theta 旋转角度
                        float(params[1]),  # phi 相位
                        float(params[2]),  # lambda 最终相位
                        qubits[0]
                    )
                else:
                    # 参数数量不匹配
                    expected = 3 if name == 'u' else 2 if name == 'u2' else 1
                    print(f"Invalid parameters for {name} gate. Expected: "
                          f"{expected} parameters, got {len(params)}")
            
            # 处理受控旋转门（crz）
            elif name in ('crz',):
                qiskit_circuit.__getattribute__(name)(
                    *[float(param) for param in params], *qubits)
            
            # 未知的门类型
            else:
                gate_name = gate['name']
                supported_gates = ['rx', 'ry', 'rz', 'cz', 'cx', 'h', 'x', 'u', 'u1', 'u2', 'crz']
                print(f'Unknown gate: {gate_name}. Supported gates: {", ".join(supported_gates)}')

        # 在每层之间添加 barrier（栅栏）
        if barrier:
            qiskit_circuit.barrier()

    return qiskit_circuit

def qiskit_to_circuit(qiskit_circuit: QuantumCircuit) -> Circuit:
    """
    将 Qiskit QuantumCircuit 转换为本框架的 Circuit 格式
    
    通过 DAG（有向无环图）分析 Qiskit 电路的依赖关系，
    将其转换为分层的电路表示。
    
    Args:
        qiskit_circuit (QuantumCircuit): Qiskit 量子电路
    
    Returns:
        Circuit: 转换后的电路对象
    """
    # 获取分层的指令列表
    layer_to_qiskit_instructions = _get_layered_instructions(qiskit_circuit)[0]

    # 将每层的 Qiskit 指令转换为本框架的门格式
    layer_to_instructions = []
    for layer_instructions in layer_to_qiskit_instructions:
        # 将每个指令转换为门字典
        layer_instructions = [_instruction_to_gate(
            instruction) for instruction in layer_instructions]
        layer_to_instructions.append(layer_instructions)

    # 创建并返回 Circuit 对象
    return Circuit(layer_to_instructions, n_qubits=qiskit_circuit.num_qubits)


def _instruction_to_gate(instruction: Instruction):
    """
    将 Qiskit 指令转换为本框架的门字典格式
    
    Args:
        instruction (Instruction): Qiskit 指令对象
    
    Returns:
        dict: 门字典，包含 'name'、'qubits' 和 'params' 键
    
    Example:
        instruction -> {
            'name': 'rx',
            'qubits': [0],
            'params': [1.5707963]
        }
    """
    # 获取门的名称
    name = instruction.operation.name
    
    # 获取门的参数
    parms = list(instruction.operation.params)
    
    # 构建门字典
    return {
        'name': name,
        'qubits': [qubit.index for qubit in instruction.qubits],  # 提取量子比特索引
        'params': [ele if isinstance(ele, float) else float(ele) for ele in parms],  # 确保参数为浮点数
    }


def _get_layered_instructions(circuit: QuantumCircuit):
    """
    获取 Qiskit 电路的分层指令
    
    通过 DAG 分析，将电路中的指令按照依赖关系分层。
    同一层中的指令可以并行执行（不存在依赖关系）。
    
    Args:
        circuit (QuantumCircuit): Qiskit 量子电路
    
    Returns:
        tuple: (layer2instructions, instruction2layer, instructions, dagcircuit, nodes)
            - layer2instructions: 分层的指令列表
            - instruction2layer: 指令到层的映射
            - instructions: 所有指令列表
            - dagcircuit: DAG 电路对象
            - nodes: DAG 节点列表
    """
    # 将 Qiskit 电路转换为 DAG 表示
    dagcircuit, instructions, nodes = _circuit_to_dag(circuit)
    
    # 获取 DAG 的多图层
    graph_layers = dagcircuit.multigraph_layers()

    # 提取操作节点，移除输入、输出和 barrier/measure 节点
    layer2operations = []
    for layer in graph_layers:
        # 过滤出实际的操作节点，排除 barrier 和 measure
        layer = [node for node in layer if isinstance(
            node, DAGOpNode) and node.op.name not in ('barrier', 'measure')]
        if len(layer) != 0:
            layer2operations.append(layer)

    # 构建指令到层的映射
    layer2instructions = []
    instruction2layer = [None] * len(nodes)
    
    for layer, operations in enumerate(layer2operations):
        layer_instructions = []
        for node in operations:
            assert node.op.name != 'barrier'
            # 找到该节点对应的指令索引
            index = nodes.index(node)
            layer_instructions.append(instructions[index])
            # 记录该指令所在的层
            instruction2layer[index] = layer
        layer2instructions.append(layer_instructions)

    return layer2instructions, instruction2layer, instructions, dagcircuit, nodes


def _circuit_to_dag(circuit: QuantumCircuit):
    """
    将 Qiskit QuantumCircuit 转换为 DAG（有向无环图）表示
    
    DAG 表示能够清晰地表达量子门之间的依赖关系，
    便于进行电路优化和分层分析。
    
    Args:
        circuit (QuantumCircuit): Qiskit 量子电路
    
    Returns:
        tuple: (dagcircuit, instructions, dagnodes)
            - dagcircuit: DAG 电路对象
            - instructions: 指令列表（不包括 barrier）
            - dagnodes: DAG 节点列表
    """
    instructions = []
    dagnodes = []

    # 创建 DAG 电路对象
    dagcircuit = DAGCircuit()
    
    # 复制电路的元数据
    dagcircuit.name = circuit.name if 'name' in circuit else None
    dagcircuit.global_phase = circuit.global_phase
    dagcircuit.calibrations = circuit.calibrations
    dagcircuit.metadata = circuit.metadata

    # 添加量子比特和经典比特
    dagcircuit.add_qubits(circuit.qubits)
    dagcircuit.add_clbits(circuit.clbits)

    # 添加量子寄存器和经典寄存器
    for register in circuit.qregs:
        dagcircuit.add_qreg(register)

    for register in circuit.cregs:
        dagcircuit.add_creg(register)

    # 将电路中的每个指令转换为 DAG 节点
    for instruction in circuit.data:
        operation = instruction.operation

        # 将操作添加到 DAG
        dag_node = dagcircuit.apply_operation_back(
            operation, instruction.qubits, instruction.clbits
        )
        
        # 跳过 barrier 指令
        if operation.name == 'barrier':
            continue
        
        # 记录指令和对应的 DAG 节点
        instructions.append(instruction)
        dagnodes.append(dag_node)
        
        # 验证指令和节点的操作名称一致
        assert instruction.operation.name == dag_node.op.name

    # 复制电路的时间信息
    dagcircuit.duration = circuit.duration
    dagcircuit.unit = circuit.unit
    
    return dagcircuit, instructions, dagnodes


def assign_barrier(qiskit_circuit):
    """
    为 Qiskit 电路的每一层之间添加 barrier（栅栏）
    
    通过 DAG 分析确定电路的层结构，然后在每层之间插入 barrier。
    barrier 用于标记电路中的逻辑分界点，便于可视化和优化。
    
    Args:
        qiskit_circuit (QuantumCircuit): 输入的 Qiskit 量子电路
    
    Returns:
        QuantumCircuit: 添加了 barrier 的新电路
    """
    # 获取分层的指令
    layer2instructions, instruction2layer, instructions, dagcircuit, nodes = _get_layered_instructions(
        qiskit_circuit) 

    # 创建新的电路
    new_circuit = QuantumCircuit(qiskit_circuit.num_qubits)
    
    # 遍历每一层，添加指令和 barrier
    for layer, instructions in enumerate(layer2instructions):
        # 添加该层的所有指令
        for instruction in instructions:
            new_circuit.append(instruction)
        # 在层之间添加 barrier
        new_circuit.barrier()

    return new_circuit
