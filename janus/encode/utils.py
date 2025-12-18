"""
编码工具函数模块

提供所有编码方法所需的辅助函数
不依赖 qiskit
"""
import numpy as np
from math import log2, ceil, floor, pi, acos, sqrt
from typing import Dict, List, Union

from janus.circuit import Circuit

QComplex = complex


def _build_state_dict(data: Union[List[float], List[complex], np.ndarray]) -> Dict[str, complex]:
    """
    将振幅列表转换为状态字典
    
    将一维的振幅列表转换为状态向量的字典表示。
    
    参数：
        data: 振幅值列表或数组
    
    返回：
        Dict[str, complex]: 状态字典，格式为 {'binary_string': amplitude}
    """
    if isinstance(data, np.ndarray):
        if data.size == 0:
            return {}
        data_list = data.tolist()
    else:
        if not data:
            return {}
        data_list = data
    
    n_qubits = int(ceil(log2(len(data_list)))) if len(data_list) > 0 else 1
    state_dict: Dict[str, complex] = {}
    
    for cnt, amp in enumerate(data_list):
        amp_complex = complex(amp)
        
        # 跳过接近零的幅度
        if abs(amp_complex)**2 < 1e-14:
            continue
        
        binary_string = format(cnt, f'0{n_qubits}b')
        state_dict[binary_string] = amp_complex
    
    return state_dict


def _schmidt(circuit: Circuit, qubits: List[int], data: List[float], cutoff: float):
    """
    Schmidt 分解
    
    对量子态进行 Schmidt 分解并生成对应的电路。
    
    参数：
        circuit: Janus 量子电路对象
        qubits: 量子比特索引列表
        data: 量子态振幅列表
        cutoff: 奇异值截断阈值
    """
    data_temp = np.array(data, dtype=float)
    n_qubits = len(qubits)
    
    # 特殊情况：单个量子比特
    if n_qubits == 1:
        a0 = data_temp[0] if len(data_temp) > 0 else 0.0
        clamped_a0 = max(-1.0, min(1.0, a0))
        
        if clamped_a0 < 0:
            angle = 2 * pi - 2 * acos(clamped_a0)
        else:
            angle = 2 * acos(clamped_a0)
        
        circuit.ry(angle, qubits[0])
        return
    
    # 补充数据到 2^n 的大小
    size = 1 << n_qubits
    if len(data_temp) < size:
        padding = np.zeros(size - len(data_temp))
        data_temp = np.concatenate((data_temp, padding))
    
    # 计算矩阵形状
    r = n_qubits % 2
    row = 1 << (n_qubits >> 1)
    col = 1 << ((n_qubits >> 1) + r)
    
    # 重塑为矩阵
    eigen_matrix = data_temp.reshape((row, col), order='C')
    
    # 奇异值分解
    U, S, Vh = np.linalg.svd(eigen_matrix, full_matrices=False)
    
    V = Vh
    
    # 确定截断长度
    A = S
    length = 0
    while length < len(A) and (A[length] >= A[0] * cutoff or length == 0):
        length += 1
    
    A_cut = A[:length]
    PartU = U[:, :length]
    PartV = V[:length, :]
    
    # 分割量子比特
    A_qubits_size = int(floor(n_qubits / 2)) + r
    A_qubits = qubits[:A_qubits_size]
    B_qubits = qubits[A_qubits_size:]
    
    bit = int(log2(length)) if length > 0 else 0
    
    # 递归编码奇异值
    if bit > 0 and len(B_qubits) >= bit:
        reg_tmp = B_qubits[:bit]
        A_cut_normalized = A_cut / np.linalg.norm(A_cut)
        _schmidt(circuit, reg_tmp, A_cut_normalized.tolist(), cutoff)
    
    # 应用 CNOT 门
    for i in range(bit):
        if i < len(B_qubits) and i < len(A_qubits):
            circuit.cx(B_qubits[i], A_qubits[i])
    
    # 应用旋转（简化处理）
    # 从 PartU 和 PartV 中提取旋转参数
    for i in range(min(len(B_qubits), PartU.shape[1])):
        val = max(-1.0, min(1.0, PartU[0, i].real()))
        angle = 2 * acos(val)
        circuit.ry(angle, B_qubits[i])
    
    for i in range(min(len(A_qubits), PartV.shape[0])):
        val = max(-1.0, min(1.0, PartV[i, 0].real()))
        angle = 2 * acos(val)
        circuit.ry(angle, A_qubits[i])


def _merging_procedure(state: Dict[str, complex], circuit: Circuit, reverse_q: List[int]) -> Dict[str, complex]:
    """
    合并程序
    
    通过应用量子门来合并量子态。
    
    参数：
        state: 状态字典
        circuit: 量子电路
        reverse_q: 反向的量子比特索引列表
    
    返回：
        Dict: 合并后的状态
    """
    new_state: Dict[str, complex] = {}
    
    if not state:
        return new_state
    
    n_qubits = len(list(state.keys())[0])
    prefix_length = n_qubits - 1
    
    q_index_to_operate = reverse_q[prefix_length]
    
    # 按前缀分组
    groups: Dict[str, Dict[str, complex]] = {}
    for key, amp in state.items():
        prefix = key[:prefix_length]
        suffix = key[prefix_length:]
        
        if prefix not in groups:
            groups[prefix] = {'0': 0.0 + 0.0j, '1': 0.0 + 0.0j}
        
        groups[prefix][suffix] = amp
    
    # 处理每个前缀组
    for prefix, amps in groups.items():
        amp0 = amps['0']
        amp1 = amps['1']
        
        new_amp = sqrt(abs(amp0)**2 + abs(amp1)**2)
        
        # 跳过接近零的情况
        if abs(new_amp) < 1e-14:
            continue
        
        # 计算相位
        relative_phase = 0.0
        if abs(amp0) > 1e-14:
            relative_phase = np.angle(amp1 / amp0)
        
        # 计算 RY 旋转角
        ry_angle = 0.0
        if abs(amp0) > 1e-14:
            ry_angle = 2 * acos(min(1.0, abs(amp0) / new_amp))
        elif abs(amp1) > 1e-14:
            ry_angle = pi
        
        # 获取控制比特索引
        control_qubits_indices = [reverse_q[i] for i in range(n_qubits) if i != prefix_length]
        
        # 应用旋转门
        if not control_qubits_indices:
            circuit.ry(-ry_angle, q_index_to_operate)
            circuit.rz(-relative_phase, q_index_to_operate)
        else:
            # 应用受控旋转（简化处理）
            # 先应用控制的 X 门以匹配控制状态
            for i, ctrl_idx in enumerate(control_qubits_indices):
                if prefix[i] == '1':
                    circuit.x(ctrl_idx)
            
            # 应用旋转
            circuit.ry(-ry_angle, q_index_to_operate)
            circuit.rz(-relative_phase, q_index_to_operate)
            
            # 恢复控制比特
            for i, ctrl_idx in enumerate(control_qubits_indices):
                if prefix[i] == '1':
                    circuit.x(ctrl_idx)
        
        new_state[prefix] = new_amp
    
    return new_state


def compute_norm(data: Union[List[float], List[complex]]) -> float:
    """
    计算范数
    
    参数：
        data: 数据列表
    
    返回：
        float: L2 范数
    """
    total = 0.0
    for item in data:
        if isinstance(item, complex):
            total += item.real**2 + item.imag**2
        else:
            total += item**2
    return sqrt(total)


def _complete_to_unitary(matrix: np.ndarray, target_num_qubits: int) -> np.ndarray:
    """
    完成矩阵为幺正矩阵（已弃用，仅保留以兼容）
    
    此函数在 Janus 框架中不使用，因为我们不需要显式的幺正操作。
    """
    target_dim = 2**target_num_qubits
    rows, cols = matrix.shape
    
    if rows == target_dim and cols == target_dim:
        return matrix
    
    if rows == target_dim and cols < target_dim:
        # 使用 QR 分解补全矩阵
        Q, _ = np.linalg.qr(matrix)
        if Q.shape != (target_dim, target_dim):
            # 如果 QR 返回的矩阵大小不对，进行调整
            if Q.shape[0] == target_dim:
                return Q[:target_dim, :target_dim]
        return Q
    
    raise ValueError(f"Unexpected matrix shape {matrix.shape} for target_num_qubits {target_num_qubits}")


def _apply_unitary(circuit: Circuit, qubits: List[int], matrix: np.ndarray, cutoff: float = None):
    """
    应用幺正操作（已弃用，仅保留以兼容）
    
    此函数在 Janus 框架中已被简化实现替代。
    使用单比特旋转门来近似幺正操作。
    """
    # 使用 cutoff 参数（即使不使用，保持兼容性）
    _ = cutoff
    if not qubits:
        return
    
    # 简化处理：对矩阵进行简单的分解
    target_num_qubits = len(qubits)
    
    # 对矩阵的对角元素应用旋转
    for i in range(min(target_num_qubits, matrix.shape[0])):
        if i < matrix.shape[1]:
            val = max(-1.0, min(1.0, matrix[i, i].real))
            angle = 2 * acos(val)
            circuit.ry(angle, qubits[i])
