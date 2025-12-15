# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
KAK Decomposition for any number of qubits.
"""
from __future__ import annotations

import numpy as np
from qiskit.circuit import QuantumCircuit, Gate


def decompose_kak(
    unitary_or_gate,
    fidelity: float = 1.0 - 1.0e-9,
    euler_basis: str = 'ZXZ',
    simplify: bool = False,
    atol: float = 1e-12,
):
    """
    Decompose a unitary matrix or gate using KAK decomposition for any number of qubits.

    For 1-qubit: Uses Euler angle decomposition.
    For 2-qubit: Uses Cartan KAK decomposition (TwoQubitWeylDecomposition).
    For n-qubit (n >= 3): Uses Quantum Shannon Decomposition (QSD).

    Args:
        unitary_or_gate: A unitary matrix or a gate object to decompose.
        fidelity: The target fidelity of the decomposition. Default is 1.0 - 1.0e-9.
        euler_basis: The basis to use for 1-qubit rotations. Default is 'ZXZ'.
            For 2-qubit decomposition, this parameter is passed to TwoQubitWeylDecomposition.
            For n-qubit decomposition, this parameter is passed to the 1-qubit decomposer in QSD.
        simplify: Whether to simplify the decomposed circuit. Default is False.
        atol: The absolute tolerance for checking zero values. Default is 1e-12.

    Returns:
        QuantumCircuit: The decomposed circuit.

    Raises:
        ValueError: If the input is not a valid unitary matrix or gate.
    """
    # Determine the dimension and number of qubits
    if isinstance(unitary_or_gate, Gate):
        # Handle Gate object
        num_qubits = unitary_or_gate.num_qubits
        from qiskit.quantum_info import Operator
        unitary_matrix = Operator(unitary_or_gate).data
        dim = unitary_matrix.shape[0]
    else:
        # Handle unitary matrix
        unitary_matrix = np.asarray(unitary_or_gate)
        if unitary_matrix.ndim != 2 or unitary_matrix.shape[0] != unitary_matrix.shape[1]:
            raise ValueError("Input must be a square matrix or a Gate object.")
        dim = unitary_matrix.shape[0]
        if dim & (dim - 1) != 0:  # Check if dim is a power of 2
            raise ValueError("Input matrix dimension must be a power of 2.")
        num_qubits = int(np.log2(dim))

    # Check if the matrix is unitary
    if not np.allclose(unitary_matrix @ unitary_matrix.conj().T, np.eye(dim)):
        # Try to find the closest unitary matrix
        from scipy.linalg import svd
        U, _, Vh = svd(unitary_matrix)
        unitary_matrix = U @ Vh

    # Choose decomposition method based on the number of qubits
    if num_qubits == 1:
        # 1-qubit: Euler angle decomposition
        from qiskit.synthesis.one_qubit import decompose_one_qubit
        return decompose_one_qubit(unitary_matrix, basis=euler_basis, simplify=simplify, atol=atol)
    
    elif num_qubits == 2:
        # 2-qubit: KAK decomposition using TwoQubitWeylDecomposition
        from qiskit.synthesis.two_qubit.two_qubit_decompose import TwoQubitWeylDecomposition
        
        weyl_decomp = TwoQubitWeylDecomposition(
            unitary_matrix,
            fidelity=fidelity
        )
        # Pass euler_basis, simplify, and atol parameters to the circuit method
        return weyl_decomp.circuit(euler_basis=euler_basis, simplify=simplify, atol=atol)
    
    else:
        # n-qubit (n >= 3): Quantum Shannon Decomposition (QSD)
        from qiskit.synthesis.unitary.qsd import qs_decomposition
        from qiskit.synthesis.one_qubit import OneQubitEulerDecomposer
        
        # Create 1-qubit decomposer with the specified basis
        decomposer_1q = OneQubitEulerDecomposer(basis=euler_basis)
        
        # Use QSD for decomposition
        return qs_decomposition(
            unitary_matrix,
            decomposer_1q=decomposer_1q,
            opt_a1=True,
            opt_a2=True
        )