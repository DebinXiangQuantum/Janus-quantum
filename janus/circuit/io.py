"""
Janus Circuit 文件读写模块

从分层格式 JSON 文件加载电路
格式: [[{'name': 'h', 'qubits': [0], 'params': []}], ...]
"""
import json
from pathlib import Path
from typing import List

from .circuit import Circuit

# 默认电路文件存储目录（项目根目录下的 circuits 文件夹）
CIRCUITS_DIR = Path(__file__).parent.parent.parent / 'circuits'


def get_circuits_dir() -> Path:
    """获取电路文件存储目录"""
    return CIRCUITS_DIR


def list_circuits() -> List[str]:
    """列出所有电路文件"""
    if not CIRCUITS_DIR.exists():
        return []
    return [f.name for f in CIRCUITS_DIR.glob('*.json')]


def load_circuit(name: str = None, filepath: str = None, n_qubits: int = None, n_clbits: int = 0) -> Circuit:
    """
    从分层格式 JSON 文件加载电路
    
    Args:
        name: 文件名（从默认目录加载）
        filepath: 完整文件路径
        n_qubits: 量子比特数（可选，自动推断）
        n_clbits: 经典比特数（默认 0）
    
    Returns:
        Circuit 实例
    
    Example:
        qc = load_circuit(name='bell')
        qc = load_circuit(filepath='./my_circuit.json')
    """
    if filepath is None:
        if name is None:
            raise ValueError("必须指定 name 或 filepath")
        if not name.endswith('.json'):
            name += '.json'
        filepath = CIRCUITS_DIR / name
    
    with open(filepath, 'r', encoding='utf-8') as f:
        layers = json.load(f)
    
    return Circuit.from_layers(layers, n_qubits=n_qubits, n_clbits=n_clbits)
