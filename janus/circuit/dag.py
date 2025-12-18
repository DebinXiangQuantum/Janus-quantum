"""
Janus DAG (有向无环图) 电路表示

DAG 表示便于进行电路优化和分析，支持：
- 基本 DAG 操作（添加、删除、替换节点）
- 拓扑排序和分层
- 祖先/后代查询
- 块收集、分割和合并
- 交换性分析 (DAGDependency)
"""
from typing import List, Dict, Set, Optional, Iterator, Tuple, Callable, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from copy import deepcopy
import numpy as np


class NodeType(Enum):
    """DAG 节点类型"""
    INPUT = "input"      # 输入节点（量子比特初始状态）
    OUTPUT = "output"    # 输出节点（量子比特最终状态）
    OP = "op"            # 操作节点（量子门）


@dataclass
class DAGNode:
    """
    DAG 节点基类
    
    Attributes:
        node_id: 节点唯一标识
        node_type: 节点类型
        qubits: 关联的量子比特
        clbits: 关联的经典比特
        op: 操作（仅 OP 类型节点）
    """
    node_id: int
    node_type: NodeType
    qubits: List[int]
    clbits: List[int] = None
    op: 'Gate' = None
    
    def __post_init__(self):
        if self.clbits is None:
            self.clbits = []
    
    @property
    def name(self) -> str:
        if self.node_type == NodeType.INPUT:
            return f"input_q{self.qubits[0]}"
        elif self.node_type == NodeType.OUTPUT:
            return f"output_q{self.qubits[0]}"
        else:
            return self.op.name if self.op else "unknown"
    
    @property
    def qargs(self) -> List[int]:
        return self.qubits
    
    @property
    def cargs(self) -> List[int]:
        return self.clbits
    
    def copy(self) -> 'DAGNode':
        """创建节点的深拷贝"""
        return DAGNode(
            node_id=self.node_id,
            node_type=self.node_type,
            qubits=self.qubits.copy(),
            clbits=self.clbits.copy() if self.clbits else [],
            op=self.op.copy() if self.op else None
        )
    
    def __repr__(self) -> str:
        if self.node_type == NodeType.OP:
            return f"DAGOpNode({self.op}, qubits={self.qubits})"
        return f"DAGNode({self.node_type.value}, qubits={self.qubits})"
    
    def __hash__(self) -> int:
        return hash(self.node_id)
    
    def __eq__(self, other) -> bool:
        if isinstance(other, DAGNode):
            return self.node_id == other.node_id
        return False


# 类型别名，便于区分不同类型的节点
DAGOpNode = DAGNode
DAGInNode = DAGNode
DAGOutNode = DAGNode


class DAGCircuit:
    """
    DAG 电路表示
    
    将量子电路表示为有向无环图，其中：
    - 节点表示量子操作
    - 边表示量子比特的数据流
    
    Attributes:
        n_qubits: 量子比特数
        n_clbits: 经典比特数
    """
    
    def __init__(self, n_qubits: int = 0, n_clbits: int = 0):
        self._n_qubits = n_qubits
        self._n_clbits = n_clbits
        
        # 节点存储
        self._nodes: Dict[int, DAGNode] = {}
        self._next_node_id = 0
        
        # 边存储: node_id -> set of successor node_ids
        self._successors: Dict[int, Set[int]] = {}
        self._predecessors: Dict[int, Set[int]] = {}
        
        # 输入输出节点
        self._input_nodes: Dict[int, DAGNode] = {}   # qubit -> input node
        self._output_nodes: Dict[int, DAGNode] = {}  # qubit -> output node
        
        # 每个量子比特当前的最后一个节点
        self._qubit_last_node: Dict[int, int] = {}
        
        # 初始化输入输出节点
        self._init_io_nodes()
    
    def _init_io_nodes(self):
        """初始化输入输出节点"""
        for q in range(self._n_qubits):
            # 输入节点
            input_node = self._create_node(NodeType.INPUT, [q])
            self._input_nodes[q] = input_node
            self._qubit_last_node[q] = input_node.node_id
            
            # 输出节点
            output_node = self._create_node(NodeType.OUTPUT, [q])
            self._output_nodes[q] = output_node
    
    def _create_node(self, node_type: NodeType, qubits: List[int], 
                     clbits: List[int] = None, op=None) -> DAGNode:
        """创建新节点"""
        node = DAGNode(
            node_id=self._next_node_id,
            node_type=node_type,
            qubits=qubits,
            clbits=clbits,
            op=op
        )
        self._nodes[node.node_id] = node
        self._successors[node.node_id] = set()
        self._predecessors[node.node_id] = set()
        self._next_node_id += 1
        return node
    
    def _add_edge(self, from_node: int, to_node: int):
        """添加边"""
        self._successors[from_node].add(to_node)
        self._predecessors[to_node].add(from_node)
    
    def _remove_edge(self, from_node: int, to_node: int):
        """移除边"""
        self._successors[from_node].discard(to_node)
        self._predecessors[to_node].discard(from_node)
    
    @property
    def n_qubits(self) -> int:
        return self._n_qubits
    
    @property
    def n_clbits(self) -> int:
        return self._n_clbits
    
    def apply_operation(self, op, qubits: List[int], clbits: List[int] = None) -> DAGNode:
        """
        添加一个操作到 DAG
        
        Args:
            op: 量子操作（Gate）
            qubits: 作用的量子比特
            clbits: 作用的经典比特
        
        Returns:
            创建的 DAGNode
        """
        # 创建操作节点
        node = self._create_node(NodeType.OP, qubits, clbits, op)
        
        # 连接前驱节点
        for q in qubits:
            last_node_id = self._qubit_last_node[q]
            self._add_edge(last_node_id, node.node_id)
            self._qubit_last_node[q] = node.node_id
        
        return node
    
    def finalize(self):
        """完成 DAG 构建，连接到输出节点"""
        for q in range(self._n_qubits):
            last_node_id = self._qubit_last_node[q]
            output_node = self._output_nodes[q]
            self._add_edge(last_node_id, output_node.node_id)
    
    def op_nodes(self) -> Iterator[DAGNode]:
        """迭代所有操作节点"""
        for node in self._nodes.values():
            if node.node_type == NodeType.OP:
                yield node
    
    def topological_op_nodes(self) -> Iterator[DAGNode]:
        """按拓扑顺序迭代操作节点"""
        # Kahn's algorithm
        in_degree = {nid: len(self._predecessors[nid]) for nid in self._nodes}
        queue = [nid for nid, deg in in_degree.items() if deg == 0]
        
        while queue:
            node_id = queue.pop(0)
            node = self._nodes[node_id]
            
            if node.node_type == NodeType.OP:
                yield node
            
            for succ_id in self._successors[node_id]:
                in_degree[succ_id] -= 1
                if in_degree[succ_id] == 0:
                    queue.append(succ_id)
    
    def layers(self) -> List[List[DAGNode]]:
        """
        获取 DAG 的分层表示
        
        Returns:
            每层包含可并行执行的操作节点
        """
        result = []
        qubit_layer = {q: -1 for q in range(self._n_qubits)}
        
        for node in self.topological_op_nodes():
            # 计算该节点应该在哪一层
            layer_idx = 0
            for q in node.qubits:
                layer_idx = max(layer_idx, qubit_layer[q] + 1)
            
            # 确保有足够的层
            while len(result) <= layer_idx:
                result.append([])
            
            result[layer_idx].append(node)
            
            # 更新量子比特的层
            for q in node.qubits:
                qubit_layer[q] = layer_idx
        
        return result
    
    def depth(self) -> int:
        """获取 DAG 深度"""
        return len(self.layers())
    
    def count_ops(self) -> Dict[str, int]:
        """统计各类操作的数量"""
        counts = {}
        for node in self.op_nodes():
            name = node.op.name if node.op else "unknown"
            counts[name] = counts.get(name, 0) + 1
        return counts
    
    def predecessors(self, node: DAGNode) -> Iterator[DAGNode]:
        """获取节点的前驱"""
        for pred_id in self._predecessors[node.node_id]:
            yield self._nodes[pred_id]
    
    def successors(self, node: DAGNode) -> Iterator[DAGNode]:
        """获取节点的后继"""
        for succ_id in self._successors[node.node_id]:
            yield self._nodes[succ_id]
    
    def remove_op_node(self, node: DAGNode):
        """
        移除一个操作节点，重新连接其前驱和后继
        """
        if node.node_type != NodeType.OP:
            raise ValueError("Can only remove OP nodes")
        
        # 对于每个量子比特，连接前驱到后继
        for q in node.qubits:
            # 找到该量子比特上的前驱和后继
            pred = None
            succ = None
            
            for p in self.predecessors(node):
                if q in p.qubits:
                    pred = p
                    break
            
            for s in self.successors(node):
                if q in s.qubits:
                    succ = s
                    break
            
            if pred and succ:
                self._add_edge(pred.node_id, succ.node_id)
        
        # 移除所有边
        for pred_id in list(self._predecessors[node.node_id]):
            self._remove_edge(pred_id, node.node_id)
        for succ_id in list(self._successors[node.node_id]):
            self._remove_edge(node.node_id, succ_id)
        
        # 移除节点
        del self._nodes[node.node_id]
        del self._successors[node.node_id]
        del self._predecessors[node.node_id]
    
    def substitute_node(self, node: DAGNode, new_op) -> DAGNode:
        """替换节点的操作"""
        if node.node_type != NodeType.OP:
            raise ValueError("Can only substitute OP nodes")
        node.op = new_op
        return node
    
    def ancestors(self, node: DAGNode) -> Set[DAGNode]:
        """
        获取节点的所有祖先（传递闭包）
        
        Args:
            node: 目标节点
        
        Returns:
            所有祖先节点的集合
        """
        result = set()
        stack = list(self._predecessors[node.node_id])
        
        while stack:
            current_id = stack.pop()
            if current_id not in [n.node_id for n in result]:
                current_node = self._nodes[current_id]
                result.add(current_node)
                stack.extend(self._predecessors[current_id])
        
        return result
    
    def descendants(self, node: DAGNode) -> Set[DAGNode]:
        """
        获取节点的所有后代（传递闭包）
        
        Args:
            node: 目标节点
        
        Returns:
            所有后代节点的集合
        """
        result = set()
        stack = list(self._successors[node.node_id])
        
        while stack:
            current_id = stack.pop()
            if current_id not in [n.node_id for n in result]:
                current_node = self._nodes[current_id]
                result.add(current_node)
                stack.extend(self._successors[current_id])
        
        return result
    
    def copy(self) -> 'DAGCircuit':
        """创建 DAGCircuit 的深拷贝"""
        new_dag = DAGCircuit(self._n_qubits, self._n_clbits)
        
        # 清除默认初始化的节点
        new_dag._nodes.clear()
        new_dag._successors.clear()
        new_dag._predecessors.clear()
        new_dag._input_nodes.clear()
        new_dag._output_nodes.clear()
        new_dag._qubit_last_node.clear()
        
        # 复制所有节点
        for node_id, node in self._nodes.items():
            new_dag._nodes[node_id] = node.copy()
        
        # 复制边
        for node_id, succs in self._successors.items():
            new_dag._successors[node_id] = succs.copy()
        for node_id, preds in self._predecessors.items():
            new_dag._predecessors[node_id] = preds.copy()
        
        # 复制输入输出节点引用
        for q, node in self._input_nodes.items():
            new_dag._input_nodes[q] = new_dag._nodes[node.node_id]
        for q, node in self._output_nodes.items():
            new_dag._output_nodes[q] = new_dag._nodes[node.node_id]
        
        new_dag._qubit_last_node = self._qubit_last_node.copy()
        new_dag._next_node_id = self._next_node_id
        
        return new_dag
    
    def replace_block_with_op(self, node_block: List[DAGNode], op, 
                               wire_pos_map: Dict[int, int], cycle_check: bool = True):
        """
        用单个操作替换一组节点
        
        Args:
            node_block: 要替换的节点列表
            op: 新操作
            wire_pos_map: 量子比特位置映射
            cycle_check: 是否检查循环（暂未实现）
        """
        if not node_block:
            raise ValueError("Cannot replace empty block")
        
        # 收集块中所有量子比特
        block_qubits = set()
        block_clbits = set()
        block_ids = {n.node_id for n in node_block}
        
        for node in node_block:
            block_qubits.update(node.qubits)
            block_clbits.update(node.clbits)
        
        # 找到块的所有外部前驱和后继
        external_preds: Dict[int, Set[int]] = {}  # qubit -> pred_node_ids
        external_succs: Dict[int, Set[int]] = {}  # qubit -> succ_node_ids
        
        for node in node_block:
            for q in node.qubits:
                for pred in self.predecessors(node):
                    if pred.node_id not in block_ids and q in pred.qubits:
                        if q not in external_preds:
                            external_preds[q] = set()
                        external_preds[q].add(pred.node_id)
                
                for succ in self.successors(node):
                    if succ.node_id not in block_ids and q in succ.qubits:
                        if q not in external_succs:
                            external_succs[q] = set()
                        external_succs[q].add(succ.node_id)
        
        # 移除块中的节点
        for node in node_block:
            self.remove_op_node(node)
        
        # 创建新节点
        sorted_qubits = sorted(block_qubits, key=lambda x: wire_pos_map.get(x, x))
        sorted_clbits = sorted(block_clbits, key=lambda x: wire_pos_map.get(x, x))
        
        new_node = self._create_node(NodeType.OP, sorted_qubits, sorted_clbits, op)
        
        # 重新连接边
        for q in sorted_qubits:
            if q in external_preds:
                for pred_id in external_preds[q]:
                    self._add_edge(pred_id, new_node.node_id)
            if q in external_succs:
                for succ_id in external_succs[q]:
                    self._add_edge(new_node.node_id, succ_id)
    
    def nodes_on_wire(self, wire: int, only_ops: bool = False) -> Iterator[DAGNode]:
        """
        获取指定量子比特上的所有节点
        
        Args:
            wire: 量子比特索引
            only_ops: 是否只返回操作节点
        
        Yields:
            该量子比特上的节点
        """
        for node in self.topological_op_nodes():
            if wire in node.qubits:
                yield node
    
    def two_qubit_ops(self) -> Iterator[DAGNode]:
        """迭代所有两比特操作"""
        for node in self.op_nodes():
            if len(node.qubits) == 2:
                yield node
    
    def multi_qubit_ops(self) -> Iterator[DAGNode]:
        """迭代所有多比特操作（>= 2 量子比特）"""
        for node in self.op_nodes():
            if len(node.qubits) >= 2:
                yield node
    
    def gate_nodes(self) -> Iterator[DAGNode]:
        """迭代所有门操作节点（排除测量、重置等）"""
        non_gate_ops = {'measure', 'reset', 'barrier', 'delay'}
        for node in self.op_nodes():
            if node.op and node.op.name not in non_gate_ops:
                yield node
    
    def longest_path(self) -> List[DAGNode]:
        """
        获取 DAG 中的最长路径
        
        Returns:
            最长路径上的节点列表
        """
        # 计算每个节点的最长路径长度
        dist = {}
        parent = {}
        
        for node in self.topological_op_nodes():
            max_dist = 0
            max_parent = None
            
            for pred in self.predecessors(node):
                if pred.node_type == NodeType.OP:
                    if dist.get(pred.node_id, 0) + 1 > max_dist:
                        max_dist = dist.get(pred.node_id, 0) + 1
                        max_parent = pred
            
            dist[node.node_id] = max_dist
            parent[node.node_id] = max_parent
        
        if not dist:
            return []
        
        # 找到最长路径的终点
        end_node_id = max(dist, key=dist.get)
        
        # 回溯构建路径
        path = []
        current_id = end_node_id
        while current_id is not None:
            path.append(self._nodes[current_id])
            parent_node = parent.get(current_id)
            current_id = parent_node.node_id if parent_node else None
        
        return path[::-1]
    
    @property
    def qubits(self) -> List[int]:
        """返回量子比特列表"""
        return list(range(self._n_qubits))
    
    @property
    def clbits(self) -> List[int]:
        """返回经典比特列表"""
        return list(range(self._n_clbits))
    
    def __repr__(self) -> str:
        op_count = sum(1 for _ in self.op_nodes())
        return f"DAGCircuit(n_qubits={self._n_qubits}, ops={op_count}, depth={self.depth()})"


def circuit_to_dag(circuit: 'Circuit') -> DAGCircuit:
    """
    将 Circuit 转换为 DAGCircuit
    
    Args:
        circuit: Janus Circuit
    
    Returns:
        DAGCircuit
    """
    dag = DAGCircuit(circuit.n_qubits, circuit.n_clbits)
    
    for inst in circuit.instructions:
        dag.apply_operation(inst.operation, inst.qubits, inst.clbits)
    
    dag.finalize()
    return dag


def dag_to_circuit(dag: DAGCircuit) -> 'Circuit':
    """
    将 DAGCircuit 转换为 Circuit
    
    Args:
        dag: DAGCircuit
    
    Returns:
        Janus Circuit
    """
    from .circuit import Circuit
    
    circuit = Circuit(dag.n_qubits, dag.n_clbits)
    
    for node in dag.topological_op_nodes():
        circuit.append(node.op.copy(), node.qubits, node.clbits)
    
    return circuit


# ==================== 高级 DAG 功能 ====================

class DAGDependency:
    """
    基于操作依赖（非交换性）的 DAG 表示
    
    与 DAGCircuit 不同，DAGDependency 中的边表示两个操作不可交换。
    这对于电路优化非常有用，因为可以识别可以重新排序的操作。
    
    参考: Iten et al., 2020. https://arxiv.org/abs/1909.05270
    """
    
    def __init__(self):
        self.name = None
        self.metadata = {}
        self._nodes: List[DAGNode] = []
        self._edges: Dict[int, Set[int]] = {}  # node_id -> set of dependent node_ids
        self._reverse_edges: Dict[int, Set[int]] = {}
        self.qubits: List[int] = []
        self.clbits: List[int] = []
        self._global_phase: float = 0.0
    
    @property
    def global_phase(self) -> float:
        return self._global_phase
    
    @global_phase.setter
    def global_phase(self, angle: float):
        self._global_phase = angle % (2 * np.pi)
    
    def size(self) -> int:
        """返回操作节点数量"""
        return len(self._nodes)
    
    def depth(self) -> int:
        """返回 DAG 深度（最长路径）"""
        if not self._nodes:
            return 0
        
        # 使用动态规划计算最长路径
        depths = {}
        
        def get_depth(node_id: int) -> int:
            if node_id in depths:
                return depths[node_id]
            
            preds = self._reverse_edges.get(node_id, set())
            if not preds:
                depths[node_id] = 1
            else:
                depths[node_id] = 1 + max(get_depth(p) for p in preds)
            return depths[node_id]
        
        return max(get_depth(n.node_id) for n in self._nodes)
    
    def add_qubits(self, qubits: List[int]):
        """添加量子比特"""
        self.qubits.extend(qubits)
    
    def add_clbits(self, clbits: List[int]):
        """添加经典比特"""
        self.clbits.extend(clbits)
    
    def add_op_node(self, op, qargs: List[int], cargs: List[int] = None):
        """
        添加操作节点并更新依赖边
        
        Args:
            op: 量子操作
            qargs: 量子比特列表
            cargs: 经典比特列表
        """
        node_id = len(self._nodes)
        node = DAGNode(
            node_id=node_id,
            node_type=NodeType.OP,
            qubits=qargs,
            clbits=cargs or [],
            op=op
        )
        self._nodes.append(node)
        self._edges[node_id] = set()
        self._reverse_edges[node_id] = set()
        
        # 检查与之前所有节点的交换性
        for prev_node in self._nodes[:-1]:
            if not self._commutes(prev_node, node):
                # 检查是否可达（避免添加冗余边）
                if not self._is_reachable(prev_node.node_id, node_id):
                    self._add_edge(prev_node.node_id, node_id)
    
    def _commutes(self, node1: DAGNode, node2: DAGNode) -> bool:
        """
        检查两个操作是否可交换
        
        简化实现：如果两个操作作用在不相交的量子比特上，则可交换
        """
        qubits1 = set(node1.qubits)
        qubits2 = set(node2.qubits)
        
        # 如果量子比特不相交，则可交换
        if not qubits1.intersection(qubits2):
            return True
        
        # TODO: 更复杂的交换性检查（如对角门之间的交换）
        return False
    
    def _is_reachable(self, from_id: int, to_id: int) -> bool:
        """检查从 from_id 是否可达 to_id"""
        visited = set()
        stack = [from_id]
        
        while stack:
            current = stack.pop()
            if current == to_id:
                return True
            if current in visited:
                continue
            visited.add(current)
            stack.extend(self._edges.get(current, set()))
        
        return False
    
    def _add_edge(self, from_id: int, to_id: int):
        """添加依赖边"""
        self._edges[from_id].add(to_id)
        self._reverse_edges[to_id].add(from_id)
    
    def get_nodes(self) -> Iterator[DAGNode]:
        """迭代所有节点"""
        return iter(self._nodes)
    
    def get_node(self, node_id: int) -> DAGNode:
        """获取指定节点"""
        return self._nodes[node_id]
    
    def direct_successors(self, node_id: int) -> List[int]:
        """获取直接后继节点 ID"""
        return sorted(self._edges.get(node_id, set()))
    
    def direct_predecessors(self, node_id: int) -> List[int]:
        """获取直接前驱节点 ID"""
        return sorted(self._reverse_edges.get(node_id, set()))
    
    def successors(self, node_id: int) -> List[int]:
        """获取所有后继节点 ID（传递闭包）"""
        result = set()
        stack = list(self._edges.get(node_id, set()))
        
        while stack:
            current = stack.pop()
            if current not in result:
                result.add(current)
                stack.extend(self._edges.get(current, set()))
        
        return sorted(result)
    
    def predecessors(self, node_id: int) -> List[int]:
        """获取所有前驱节点 ID（传递闭包）"""
        result = set()
        stack = list(self._reverse_edges.get(node_id, set()))
        
        while stack:
            current = stack.pop()
            if current not in result:
                result.add(current)
                stack.extend(self._reverse_edges.get(current, set()))
        
        return sorted(result)
    
    def topological_nodes(self) -> Iterator[DAGNode]:
        """按拓扑顺序迭代节点"""
        in_degree = {n.node_id: len(self._reverse_edges.get(n.node_id, set())) 
                     for n in self._nodes}
        queue = [nid for nid, deg in in_degree.items() if deg == 0]
        
        while queue:
            node_id = queue.pop(0)
            yield self._nodes[node_id]
            
            for succ_id in self._edges.get(node_id, set()):
                in_degree[succ_id] -= 1
                if in_degree[succ_id] == 0:
                    queue.append(succ_id)
    
    def copy(self) -> 'DAGDependency':
        """创建 DAGDependency 的深拷贝"""
        dag = DAGDependency()
        dag.name = self.name
        dag.metadata = self.metadata.copy()
        dag.qubits = self.qubits.copy()
        dag.clbits = self.clbits.copy()
        dag._global_phase = self._global_phase
        dag._nodes = [n.copy() for n in self._nodes]
        dag._edges = {k: v.copy() for k, v in self._edges.items()}
        dag._reverse_edges = {k: v.copy() for k, v in self._reverse_edges.items()}
        return dag
    
    def replace_block_with_op(self, node_block: List[DAGNode], op, 
                               wire_pos_map: Dict[int, int], cycle_check: bool = True):
        """
        用单个操作替换一组节点
        
        Args:
            node_block: 要替换的节点列表
            op: 新操作
            wire_pos_map: 量子比特位置映射
            cycle_check: 是否检查循环
        """
        if not node_block:
            raise ValueError("Cannot replace empty block")
        
        # 收集块中所有量子比特
        block_qubits = set()
        block_clbits = set()
        block_ids = {n.node_id for n in node_block}
        
        for node in node_block:
            block_qubits.update(node.qubits)
            block_clbits.update(node.clbits)
        
        # 找到块的所有前驱和后继
        all_preds = set()
        all_succs = set()
        
        for node in node_block:
            for pred_id in self._reverse_edges.get(node.node_id, set()):
                if pred_id not in block_ids:
                    all_preds.add(pred_id)
            for succ_id in self._edges.get(node.node_id, set()):
                if succ_id not in block_ids:
                    all_succs.add(succ_id)
        
        # 移除块中的节点
        for node in node_block:
            # 移除所有相关边
            for pred_id in list(self._reverse_edges.get(node.node_id, set())):
                self._edges[pred_id].discard(node.node_id)
            for succ_id in list(self._edges.get(node.node_id, set())):
                self._reverse_edges[succ_id].discard(node.node_id)
        
        # 创建新节点
        new_node_id = len(self._nodes)
        new_node = DAGNode(
            node_id=new_node_id,
            node_type=NodeType.OP,
            qubits=sorted(block_qubits, key=lambda x: wire_pos_map.get(x, x)),
            clbits=sorted(block_clbits, key=lambda x: wire_pos_map.get(x, x)),
            op=op
        )
        self._nodes.append(new_node)
        self._edges[new_node_id] = set()
        self._reverse_edges[new_node_id] = set()
        
        # 连接新节点
        for pred_id in all_preds:
            self._edges[pred_id].add(new_node_id)
            self._reverse_edges[new_node_id].add(pred_id)
        
        for succ_id in all_succs:
            self._edges[new_node_id].add(succ_id)
            self._reverse_edges[succ_id].add(new_node_id)


class BlockCollector:
    """
    DAG 块收集器
    
    实现各种策略将 DAG 划分为满足特定条件的节点块。
    支持 DAGCircuit 和 DAGDependency。
    """
    
    def __init__(self, dag: Union[DAGCircuit, DAGDependency]):
        self.dag = dag
        self._pending_nodes: List[DAGNode] = []
        self._in_degree: Dict[int, int] = {}
        self._collect_from_back = False
        self.is_dag_dependency = isinstance(dag, DAGDependency)
    
    def _setup_in_degrees(self):
        """设置每个节点的入度"""
        self._pending_nodes = []
        self._in_degree = {}
        
        for node in self._op_nodes():
            deg = len(self._direct_preds(node))
            self._in_degree[node.node_id] = deg
            if deg == 0:
                self._pending_nodes.append(node)
    
    def _op_nodes(self) -> Iterator[DAGNode]:
        """获取所有操作节点"""
        if self.is_dag_dependency:
            return self.dag.get_nodes()
        else:
            return self.dag.op_nodes()
    
    def _direct_preds(self, node: DAGNode) -> List[DAGNode]:
        """获取直接前驱"""
        if self.is_dag_dependency:
            if self._collect_from_back:
                return [self.dag.get_node(i) for i in self.dag.direct_successors(node.node_id)]
            else:
                return [self.dag.get_node(i) for i in self.dag.direct_predecessors(node.node_id)]
        else:
            if self._collect_from_back:
                return [n for n in self.dag.successors(node) if n.node_type == NodeType.OP]
            else:
                return [n for n in self.dag.predecessors(node) if n.node_type == NodeType.OP]
    
    def _direct_succs(self, node: DAGNode) -> List[DAGNode]:
        """获取直接后继"""
        if self.is_dag_dependency:
            if self._collect_from_back:
                return [self.dag.get_node(i) for i in self.dag.direct_predecessors(node.node_id)]
            else:
                return [self.dag.get_node(i) for i in self.dag.direct_successors(node.node_id)]
        else:
            if self._collect_from_back:
                return [n for n in self.dag.predecessors(node) if n.node_type == NodeType.OP]
            else:
                return [n for n in self.dag.successors(node) if n.node_type == NodeType.OP]
    
    def collect_matching_block(self, filter_fn: Callable[[DAGNode], bool],
                                max_block_width: Optional[int] = None) -> List[DAGNode]:
        """
        收集满足过滤条件的最大块
        
        Args:
            filter_fn: 过滤函数，返回 True 表示节点应被收集
            max_block_width: 块的最大宽度（量子比特数）
        
        Returns:
            收集到的节点列表
        """
        current_block = []
        current_block_qargs = set()
        unprocessed = self._pending_nodes
        self._pending_nodes = []
        
        while unprocessed:
            node = unprocessed.pop()
            
            if max_block_width is not None:
                new_qargs = current_block_qargs | set(node.qubits)
                width_ok = len(new_qargs) <= max_block_width
            else:
                new_qargs = set()
                width_ok = True
            
            if filter_fn(node) and width_ok:
                current_block.append(node)
                current_block_qargs = new_qargs
                
                for succ in self._direct_succs(node):
                    self._in_degree[succ.node_id] -= 1
                    if self._in_degree[succ.node_id] == 0:
                        unprocessed.append(succ)
            else:
                self._pending_nodes.append(node)
        
        return current_block
    
    def collect_all_matching_blocks(self, filter_fn: Callable[[DAGNode], bool],
                                     split_blocks: bool = True,
                                     min_block_size: int = 2,
                                     split_layers: bool = False,
                                     collect_from_back: bool = False,
                                     max_block_width: Optional[int] = None) -> List[List[DAGNode]]:
        """
        收集所有满足条件的块
        
        Args:
            filter_fn: 过滤函数
            split_blocks: 是否将块分割为不相交的子块
            min_block_size: 最小块大小
            split_layers: 是否将块分割为层
            collect_from_back: 是否从后向前收集
            max_block_width: 块的最大宽度
        
        Returns:
            块列表
        """
        def not_filter_fn(node):
            return not filter_fn(node)
        
        self._collect_from_back = collect_from_back
        self._setup_in_degrees()
        
        matching_blocks = []
        while self._pending_nodes:
            self.collect_matching_block(not_filter_fn, max_block_width=None)
            block = self.collect_matching_block(filter_fn, max_block_width=max_block_width)
            if block:
                matching_blocks.append(block)
        
        if split_layers:
            tmp = []
            for block in matching_blocks:
                tmp.extend(split_block_into_layers(block))
            matching_blocks = tmp
        
        if split_blocks:
            tmp = []
            for block in matching_blocks:
                tmp.extend(BlockSplitter().run(block))
            matching_blocks = tmp
        
        if collect_from_back:
            matching_blocks = [block[::-1] for block in matching_blocks[::-1]]
        
        matching_blocks = [b for b in matching_blocks if len(b) >= min_block_size]
        
        return matching_blocks


class BlockSplitter:
    """
    块分割器 - 将块分割为不相交量子比特上的子块
    
    使用并查集 (Disjoint Set Union) 数据结构
    """
    
    def __init__(self):
        self.leader = {}
        self.group = {}
    
    def find_leader(self, index: int) -> int:
        """查找领导者"""
        if index not in self.leader:
            self.leader[index] = index
            self.group[index] = []
            return index
        if self.leader[index] == index:
            return index
        self.leader[index] = self.find_leader(self.leader[index])
        return self.leader[index]
    
    def union_leaders(self, index1: int, index2: int):
        """合并两个集合"""
        leader1 = self.find_leader(index1)
        leader2 = self.find_leader(index2)
        if leader1 == leader2:
            return
        if len(self.group[leader1]) < len(self.group[leader2]):
            leader1, leader2 = leader2, leader1
        self.leader[leader2] = leader1
        self.group[leader1].extend(self.group[leader2])
        self.group[leader2].clear()
    
    def run(self, block: List[DAGNode]) -> List[List[DAGNode]]:
        """将块分割为不相交的子块"""
        for node in block:
            indices = node.qubits
            if not indices:
                continue
            first = indices[0]
            for index in indices[1:]:
                self.union_leaders(first, index)
            self.group[self.find_leader(first)].append(node)
        
        blocks = []
        for index, item in self.leader.items():
            if index == item and self.group[index]:
                blocks.append(self.group[index])
        
        return blocks


def split_block_into_layers(block: List[DAGNode]) -> List[List[DAGNode]]:
    """
    将块分割为层（深度为 1 的子块）
    
    每层中的操作不重叠，可以并行执行
    """
    bit_depths: Dict[int, int] = {}
    layers: List[List[DAGNode]] = []
    
    for node in block:
        cur_bits = set(node.qubits)
        cur_bits.update(node.clbits)
        
        cur_depth = max((bit_depths.get(bit, 0) for bit in cur_bits), default=0)
        while len(layers) <= cur_depth:
            layers.append([])
        
        for bit in cur_bits:
            bit_depths[bit] = cur_depth + 1
        layers[cur_depth].append(node)
    
    return layers


class BlockCollapser:
    """
    块合并器 - 将节点块合并为单个操作
    """
    
    def __init__(self, dag: Union[DAGCircuit, DAGDependency]):
        self.dag = dag
    
    def collapse_to_operation(self, blocks: List[List[DAGNode]], 
                               collapse_fn: Callable) -> Union[DAGCircuit, DAGDependency]:
        """
        将每个块合并为单个操作
        
        Args:
            blocks: 块列表
            collapse_fn: 合并函数，接收电路返回操作
        
        Returns:
            修改后的 DAG
        """
        from .circuit import Circuit
        
        global_index_map = {q: idx for idx, q in enumerate(
            self.dag.qubits if hasattr(self.dag, 'qubits') else range(self.dag.n_qubits)
        )}
        
        for block in blocks:
            cur_qubits = set()
            cur_clbits = set()
            
            for node in block:
                cur_qubits.update(node.qubits)
                cur_clbits.update(node.clbits)
            
            sorted_qubits = sorted(cur_qubits, key=lambda x: global_index_map.get(x, x))
            sorted_clbits = sorted(cur_clbits)
            
            # 创建子电路
            qc = Circuit(len(sorted_qubits), len(sorted_clbits))
            
            wire_pos_map = {qb: ix for ix, qb in enumerate(sorted_qubits)}
            wire_pos_map.update({cb: ix for ix, cb in enumerate(sorted_clbits)})
            
            for node in block:
                mapped_qubits = [wire_pos_map[q] for q in node.qubits]
                mapped_clbits = [wire_pos_map[c] for c in node.clbits]
                qc.append(node.op.copy(), mapped_qubits, mapped_clbits)
            
            # 合并为单个操作
            op = collapse_fn(qc)
            
            # 替换块
            if hasattr(self.dag, 'replace_block_with_op'):
                self.dag.replace_block_with_op(block, op, wire_pos_map, cycle_check=False)
        
        return self.dag


def circuit_to_dag_dependency(circuit: 'Circuit') -> DAGDependency:
    """
    将 Circuit 转换为 DAGDependency
    
    Args:
        circuit: Janus Circuit
    
    Returns:
        DAGDependency
    """
    dag = DAGDependency()
    dag.qubits = list(range(circuit.n_qubits))
    dag.clbits = list(range(circuit.n_clbits))
    
    for inst in circuit.instructions:
        dag.add_op_node(inst.operation, inst.qubits, inst.clbits)
    
    return dag


def dag_dependency_to_circuit(dag: DAGDependency) -> 'Circuit':
    """
    将 DAGDependency 转换为 Circuit
    
    Args:
        dag: DAGDependency
    
    Returns:
        Janus Circuit
    """
    from .circuit import Circuit
    
    circuit = Circuit(len(dag.qubits), len(dag.clbits))
    
    for node in dag.topological_nodes():
        circuit.append(node.op.copy(), node.qubits, node.clbits)
    
    return circuit
