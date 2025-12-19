"""
Janus 量子电路模块

提供量子电路的构建、操作和表示
"""
from .operation import Operation
from .gate import Gate, ControlledGate
from .instruction import Instruction
from .layer import Layer
from .circuit import Circuit
from .qubit import Qubit, QuantumRegister
from .clbit import Clbit, ClassicalRegister
from .parameter import Parameter, ParameterExpression
from .dag import (
    DAGCircuit, DAGNode, DAGOpNode, DAGInNode, DAGOutNode,
    circuit_to_dag, dag_to_circuit,
    DAGDependency, circuit_to_dag_dependency, dag_dependency_to_circuit,
    BlockCollector, BlockSplitter, BlockCollapser, split_block_into_layers
)

# 标准门
from .library import (
    HGate,
    XGate,
    YGate,
    ZGate,
    SGate,
    SdgGate,
    TGate,
    TdgGate,
    RXGate,
    RYGate,
    RZGate,
    UGate,
    CXGate,
    CZGate,
    CRZGate,
    SwapGate,
    Barrier,
    Measure,
    Reset,
)

__all__ = [
    # 核心类
    'Operation',
    'Gate',
    'ControlledGate',
    'Instruction',
    'Layer',
    'Circuit',
    'Qubit',
    'QuantumRegister',
    'Clbit',
    'ClassicalRegister',
    # 参数化
    'Parameter',
    'ParameterExpression',
    # DAG
    'DAGCircuit',
    'DAGNode',
    'DAGOpNode',
    'DAGInNode',
    'DAGOutNode',
    'circuit_to_dag',
    'dag_to_circuit',
    # DAGDependency (交换性分析)
    'DAGDependency',
    'circuit_to_dag_dependency',
    'dag_dependency_to_circuit',
    # 块操作
    'BlockCollector',
    'BlockSplitter',
    'BlockCollapser',
    'split_block_into_layers',
    # 标准门
    'HGate',
    'XGate',
    'YGate',
    'ZGate',
    'SGate',
    'SdgGate',
    'TGate',
    'TdgGate',
    'RXGate',
    'RYGate',
    'RZGate',
    'UGate',
    'CXGate',
    'CZGate',
    'CRZGate',
    'SwapGate',
    'Barrier',
    'Measure',
    'Reset',
]
