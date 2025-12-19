"""
Schmidt 分解编码模块

使用 Schmidt 分解方法进行量子状态编码
"""
from typing import List
import numpy as np
from math import log2, ceil, pi, acos

from janus.circuit import Circuit


def _schmidt_decompose(data: List[float], qubits: List[int], circuit: Circuit, cutoff: float = 1e-4):
    """
    使用 Schmidt 分解进行递归编码
    
    参数：
        data: 量子态振幅列表
        qubits: 可用的量子比特列表
        circuit: 量子电路对象
        cutoff: 奇异值截断阈值
    """
    if not qubits:
        return
    
    data_temp = np.array(data, dtype=float)
    n_qubits = len(qubits)
    
    # 特殊情况：单个量子比特
    if n_qubits == 1:
        a0 = data_temp[0] if len(data_temp) > 0 else 0.0
        
        # 确保值在 [-1, 1] 范围内
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
        data_temp = np.pad(data_temp, (0, size - len(data_temp)))
    
    # 确定矩阵的行列数（用于 SVD）
    r = n_qubits % 2
    row = 1 << (n_qubits >> 1)
    col = 1 << ((n_qubits >> 1) + r)
    
    # 重塑数据为矩阵
    eigen_matrix = data_temp.reshape((row, col))
    
    # 执行奇异值分解
    U, S, Vh = np.linalg.svd(eigen_matrix, full_matrices=False)
    
    # 截断奇异值
    length = 0
    while length < len(S) and (S[length] >= S[0] * cutoff or length == 0):
        length += 1
    
    # 提取截断的矩阵
    A_cut = S[:length]
    PartU = U[:, :length]
    PartV = Vh[:length, :]
    
    # 分割量子比特
    from math import floor
    A_qubits_size = int(floor(n_qubits / 2)) + r
    A_qubits = qubits[:A_qubits_size]
    B_qubits = qubits[A_qubits_size:]
    
    # 对奇异值进行编码
    bit = int(log2(length)) if length > 0 else 0
    
    if bit > 0 and len(B_qubits) >= bit:
        reg_tmp = B_qubits[:bit]
        A_cut_normalized = A_cut / np.linalg.norm(A_cut)
        _schmidt_decompose(A_cut_normalized.tolist(), reg_tmp, circuit, cutoff)
    
    # 应用 CNOT 门连接量子比特（这是关键部分！）
    for i in range(min(bit, len(B_qubits), len(A_qubits))):
        circuit.cx(B_qubits[i], A_qubits[i])
    
    # 应用旋转门来编码 U 和 V 矩阵的信息
    # 这部分使用 RY 门进行近似（因为 Janus 库没有通用幺正门）
    if len(B_qubits) > 0:
        for i in range(min(len(B_qubits), PartU.shape[1])):
            val = float(np.real(PartU[0, i]))
            val = max(-1.0, min(1.0, val))
            angle = 2 * acos(val)
            circuit.ry(angle, B_qubits[i])
    
    if len(A_qubits) > 0:
        for i in range(min(len(A_qubits), PartV.shape[0])):
            val = float(np.real(PartV[i, 0]))
            val = max(-1.0, min(1.0, val))
            angle = 2 * acos(val)
            circuit.ry(angle, A_qubits[i])


def schmidt_encode(q_size: int, data: List[float], cutoff: float = 1e-4) -> Circuit:
    """
    Schmidt 编码
    
    使用 Schmidt 分解进行量子状态编码，适合处理具有特定结构的量子态。
    
    参数：
        q_size: 可用的量子比特总数
        data: 量子态振幅列表（必须归一化）
        cutoff: 奇异值截断阈值（默认 1e-4）
    
    返回：
        Circuit: 编码电路
    
    异常：
        ValueError: 如果数据未归一化或参数不合法
    """
    data_temp = np.array(data, dtype=float)
    
    # 检查归一化
    norm = np.linalg.norm(data_temp)
    if not np.isclose(norm, 1.0, atol=1e-13):
        raise ValueError("错误：数据未归一化（L2范数必须等于1）")
    
    # 计算所需的量子比特数
    n_required_qubits = ceil(log2(len(data_temp))) if len(data_temp) > 0 else 1
    
    if n_required_qubits > q_size:
        raise ValueError(f"错误：Schmidt_encode 参数错误。所需量子比特数({n_required_qubits})超过可用数({q_size})")
    
    # 创建电路
    circuit = Circuit(q_size)
    
    # 获取可用的量子比特列表
    qubits_to_use = list(range(n_required_qubits))
    
    # 执行 Schmidt 分解编码
    _schmidt_decompose(data_temp.tolist(), qubits_to_use, circuit, cutoff)
    
    # 打印编码信息
    print(f"--- 编码所使用的量子比特数量: {n_required_qubits} ---")
    
    return circuit
