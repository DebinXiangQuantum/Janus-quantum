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
Two-qubit KAK (Cartan) decomposition interface.
"""

from __future__ import annotations

import numpy as np
from qiskit.circuit import QuantumCircuit, Gate
from qiskit.exceptions import QiskitError


def decompose_two_qubit_kak(
    unitary_or_gate, 
    fidelity: float | None = 1.0 - 1.0e-9,
    euler_basis: str | None = None,
    simplify: bool = False,
    atol: float = 1e-12
) -> QuantumCircuit:
    """
    Decompose a two-qubit unitary or gate using the Cartan KAK decomposition.
    
    This function decomposes a two-qubit unitary matrix or gate into the Cartan KAK form:
    
    .. math::
        U = ({K_1}^l \otimes {K_1}^r) e^{(i a XX + i b YY + i c ZZ)} ({K_2}^l \otimes {K_2}^r)
    
    where:
    - :math:`{K_1}^l, {K_1}^r, {K_2}^l, {K_2}^r \in SU(2)` are single-qubit unitaries
    - :math:`a, b, c` are the Weyl parameters satisfying :math:`\pi/4 \geq a \geq b \geq |c|`
    
    Args:
        unitary_or_gate: A 4x4 unitary matrix or a two-qubit Gate object to decompose.
        fidelity: Target fidelity for the decomposition. Default is 1.0 - 1.0e-9.
        euler_basis: Basis string to be used for decomposing single-qubit rotations.
            Valid options are ['ZXZ', 'ZYZ', 'XYX', 'XZX', 'U', 'U3', 'U321', 'U1X', 'PSX', 'ZSX', 'ZSXX', 'RR'].
        simplify: Whether to simplify the decomposed circuit. Default is False.
        atol: Absolute tolerance for checking zero values during simplification.
    
    Returns:
        QuantumCircuit: Decomposed quantum circuit representing the input unitary or gate.
    
    Raises:
        QiskitError: If the input is not a valid two-qubit unitary or gate.
    
    Examples:
        >>> from qiskit.circuit.library.standard_gates import decompose_two_qubit_kak
        >>> from qiskit.circuit.library import CXGate
        >>> import numpy as np
        
        >>> # Decompose a CX gate
        >>> cx_gate = CXGate()
        >>> circuit1 = decompose_two_qubit_kak(cx_gate)
        >>> print(circuit1.draw())
        
        >>> # Decompose a random two-qubit unitary
        >>> random_unitary = np.random.rand(4, 4) + 1j * np.random.rand(4, 4)
        >>> random_unitary = (random_unitary @ random_unitary.conj().T)**0.5  # Make it unitary
        >>> circuit2 = decompose_two_qubit_kak(random_unitary, euler_basis='U3', simplify=True)
        >>> print(circuit2.draw())
    """
    # Convert input to unitary matrix if it's a Gate object
    if isinstance(unitary_or_gate, Gate):
        if unitary_or_gate.num_qubits != 2:
            raise QiskitError(f"Gate must be a two-qubit gate, but got {unitary_or_gate.num_qubits}-qubit gate.")
        from qiskit.quantum_info import Operator
        unitary_matrix = Operator(unitary_or_gate).data
    elif isinstance(unitary_or_gate, np.ndarray):
        if unitary_or_gate.shape != (4, 4):
            raise QiskitError(f"Unitary matrix must be 4x4, but got shape {unitary_or_gate.shape}.")
        unitary_matrix = unitary_or_gate
    else:
        raise QiskitError(f"Input must be a two-qubit Gate or 4x4 unitary matrix, but got {type(unitary_or_gate).__name__}.")
    
    # Check if the matrix is unitary
    identity = np.eye(4)
    if not np.allclose(unitary_matrix @ unitary_matrix.conj().T, identity, atol=atol):
        raise QiskitError("Input matrix is not unitary.")
    
    # Perform KAK decomposition
    try:
        from qiskit.synthesis.two_qubit.two_qubit_decompose import TwoQubitWeylDecomposition
        kak_decomposition = TwoQubitWeylDecomposition(unitary_matrix, fidelity=fidelity)
        
        # Get the circuit representation
        circuit = kak_decomposition.circuit(euler_basis=euler_basis, simplify=simplify, atol=atol)
        
        return circuit
    except Exception as e:
        raise QiskitError(f"KAK decomposition failed: {str(e)}")