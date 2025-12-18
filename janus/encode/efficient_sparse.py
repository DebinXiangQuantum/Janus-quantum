"""
高效稀疏编码模块

使用 Janus 电路库实现高效稀疏状态编码
"""
from typing import Union, List, Dict
import numpy as np
from math import log2, ceil

from janus.circuit import Circuit


# 定义复数类型
QComplex = complex


def _build_state_dict(data: Union[List[float], List[complex], np.ndarray]) -> Dict[str, complex]:
    """
    将振幅列表转换为状态字典
    
    将一维的振幅列表转换为状态向量的字典表示，
    其中键是二进制字符串，值是对应的振幅。
    
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
    
    n_qubits = int(ceil(log2(len(data_list))))
    state_dict: Dict[str, complex] = {}
    
    for cnt, amp in enumerate(data_list):
        amp_complex = complex(amp)
        
        # 跳过接近零的幅度
        if abs(amp_complex) < 1e-14:
            continue
        
        binary_string = format(cnt, f'0{n_qubits}b')
        state_dict[binary_string] = amp_complex
    
    return state_dict


def _compute_angles(amplitude_1: complex, amplitude_2: complex) -> tuple:
    """
    计算用于合并两个振幅的旋转角度
    
    参数：
        amplitude_1: 第一个振幅
        amplitude_2: 第二个振幅
    
    返回：
        tuple: (theta, phi, lambda) 角度参数
    """
    # 计算合并后的范数
    norm = (abs(amplitude_1)**2 + abs(amplitude_2)**2)**0.5
    
    # 计算旋转角
    if abs(norm) < 1e-14:
        return (0.0, 0.0, 0.0)
    
    # 计算theta（RY旋转角）
    if abs(amplitude_1) < 1e-14:
        theta = np.pi
    else:
        cos_val = abs(amplitude_1) / norm
        cos_val = max(-1.0, min(1.0, cos_val))  # 防止数值错误
        theta = 2 * np.arccos(cos_val)
    
    # 计算相位
    if abs(amplitude_1) < 1e-14:
        phi = 0.0
    else:
        phi = -np.angle(amplitude_2 / amplitude_1)
    
    lam = 0.0
    
    return (theta, phi, lam)


def efficient_sparse(q_size: int, data: Union[List[float], List[complex], Dict[str, Union[float, complex]]]) -> Circuit:
    """
    高效稀疏编码
    
    将稀疏的量子状态高效地编码到量子电路中。
    该方法特别适用于只有少数非零振幅的状态。
    
    参数：
        q_size: 可用的量子比特总数
        data: 输入数据，可以是：
            - 振幅值列表
            - 振幅值数组
            - 状态字典 {binary_string: amplitude}
    
    返回：
        Circuit: 编码电路
    
    异常：
        TypeError: 如果输入数据类型不正确
        ValueError: 如果数据未归一化或参数不合法
    """
    # 转换输入数据为状态字典
    if isinstance(data, (list, np.ndarray)):
        state = _build_state_dict(data)
    elif isinstance(data, dict):
        state = {k: complex(v) for k, v in data.items()}
    else:
        raise TypeError("输入数据必须是振幅列表或状态字典")
    
    # 检查状态是否为空
    if not state:
        raise ValueError("错误：输入状态字典不能为空")
    
    # 检查所有状态有相同的量子比特数
    first_key = next(iter(state.keys()))
    n_qubits = len(first_key)
    
    for key in state.keys():
        if len(key) != n_qubits:
            raise ValueError("错误：输入状态的二进制字符串长度必须相同")
    
    # 检查归一化
    tmp_sum = sum(abs(amp)**2 for amp in state.values())
    max_precision = 1e-13
    
    if abs(1.0 - tmp_sum) > max_precision:
        if tmp_sum < max_precision:
            raise ValueError("错误：输入向量为零")
        raise ValueError("错误：输入向量必须满足归一化条件")
    
    # 检查量子比特数量
    if n_qubits > q_size:
        raise ValueError(f"错误：所需量子比特数({n_qubits})超过可用数({q_size})")
    
    # 创建电路
    circuit = Circuit(q_size)
    
    # 通过一系列单比特旋转和CNOT门来编码稀疏状态
    # 这是一个简化的实现，使用两比特门逐步构建状态
    
    # 获取状态的二进制表示
    state_bitstrings = list(state.keys())
    
    # 如果只有一个非零状态，直接准备该状态
    if len(state_bitstrings) == 1:
        target_state = state_bitstrings[0]
        # 对每一位应用X门如果该位是1
        for i, bit in enumerate(target_state):
            if bit == '1':
                circuit.x(i)
        return circuit
    
    # 对于多个非零状态的情况，使用合并程序递归构建
    reverse_q_indices = list(range(n_qubits - 1, -1, -1))
    current_state = state.copy()
    
    # 逐步合并状态
    for level in range(n_qubits):
        if not current_state or len(current_state) == 1:
            break
        
        # 使用合并程序进行一层的合并
        qubit_indices = reverse_q_indices[:(n_qubits - level)]
        current_state = _merging_procedure_helper(current_state, circuit, qubit_indices, q_size)
    
    # 处理最后剩余的状态
    if current_state:
        final_state = next(iter(current_state.keys()))
        # 对每一位应用X门如果该位是1
        for i, bit in enumerate(final_state):
            if bit == '1':
                circuit.x(i)
    
    return circuit


def _merging_procedure_helper(state: Dict[str, complex], circuit: Circuit, 
                             qubit_indices: List[int], q_size: int) -> Dict[str, complex]:
    """
    合并程序的辅助函数
    """
    new_state: Dict[str, complex] = {}
    
    if not state or not qubit_indices:
        return state
    
    n_qubits = len(list(state.keys())[0])
    last_qubit_idx = qubit_indices[-1]
    prefix_length = n_qubits - 1
    
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
        
        new_amp = (abs(amp0)**2 + abs(amp1)**2)**0.5
        
        if abs(new_amp) < 1e-14:
            continue
        
        # 计算旋转参数
        ry_angle = 0.0
        if abs(amp0) > 1e-14:
            ry_angle = 2 * np.arccos(min(1.0, abs(amp0) / new_amp))
        elif abs(amp1) > 1e-14:
            ry_angle = np.pi
        
        # 应用旋转
        circuit.ry(ry_angle, last_qubit_idx)
        
        new_state[prefix] = new_amp
    
    return new_state
