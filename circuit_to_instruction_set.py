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
Circuit to instruction set conversion interface.
"""
from __future__ import annotations

from typing import Optional, List, Union
from qiskit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit.transpiler import CouplingMap


def convert_circuit_to_instruction_set(
    circuit: QuantumCircuit,
    instruction_set: List[str],
    coupling_map: Optional[Union[CouplingMap, List[List[int]]]] = None,
    optimization_level: Optional[int] = 1,
    seed_transpiler: Optional[int] = None,
) -> QuantumCircuit:
    """
    Convert a quantum circuit to a specified instruction set using Qiskit's transpiler.

    Args:
        circuit: The input quantum circuit to be converted.
        instruction_set: List of basis gate names to convert to (e.g., ['u3', 'cx']).
        coupling_map: Directed coupling map to target in mapping. If None, no specific
            connectivity constraints are enforced.
        optimization_level: How much optimization to perform on the circuits.
            Higher levels generate more optimized circuits, at the expense of longer
            transpilation time.
            * 0: no optimization
            * 1: light optimization (default)
            * 2: heavy optimization
            * 3: even heavier optimization
        seed_transpiler: Sets random seed for the stochastic parts of the transpiler.

    Returns:
        QuantumCircuit: The converted circuit using only gates from the specified instruction set.

    Raises:
        ValueError: If the input is not a valid QuantumCircuit object.
        ValueError: If the instruction_set is empty or not a list of strings.

    Examples:
        Convert a circuit to use only u3 and cx gates:
        
        >>> from qiskit import QuantumCircuit
        >>> from circuit_to_instruction_set import convert_circuit_to_instruction_set
        >>> 
        >>> # Create a sample circuit
        >>> qc = QuantumCircuit(2)
        >>> qc.h(0)
        >>> qc.cx(0, 1)
        >>> qc.z(1)
        >>> 
        >>> # Convert to u3 and cx gates
        >>> converted_qc = convert_circuit_to_instruction_set(qc, ['u3', 'cx'])
        >>> print(converted_qc)
        >>> print(converted_qc.count_ops())
        
        Convert with a specific coupling map and higher optimization:
        
        >>> coupling_map = [[0, 1], [1, 2]]  # Linear chain of 3 qubits
        >>> qc3 = QuantumCircuit(3)
        >>> qc3.h(0)
        >>> qc3.cx(0, 2)  # Non-native gate for the coupling map
        >>> 
        >>> converted_qc3 = convert_circuit_to_instruction_set(
        ...     qc3, ['u3', 'cx'], coupling_map=coupling_map, optimization_level=2
        ... )
        >>> print(converted_qc3)
        >>> print(converted_qc3.count_ops())
    """
    # Input validation
    if not isinstance(circuit, QuantumCircuit):
        raise ValueError("Input must be a valid QuantumCircuit object.")
    
    if not isinstance(instruction_set, list) or not instruction_set:
        raise ValueError("instruction_set must be a non-empty list of strings.")
    
    if not all(isinstance(gate, str) for gate in instruction_set):
        raise ValueError("All elements in instruction_set must be strings.")
    
    # Use transpile to convert the circuit to the specified instruction set
    converted_circuit = transpile(
        circuit,
        basis_gates=instruction_set,
        coupling_map=coupling_map,
        optimization_level=optimization_level,
        seed_transpiler=seed_transpiler,
    )
    
    return converted_circuit


# Example usage if this file is run directly
if __name__ == "__main__":
    print("Demonstrating circuit to instruction set conversion...")
    
    # Create a sample circuit with various gates
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.z(1)
    qc.x(0)
    qc.y(1)
    
    print("Original circuit:")
    print(qc)
    print(f"Original gate counts: {qc.count_ops()}")
    
    # Convert to u3 and cx gates
    instruction_set = ['u3', 'cx']
    converted_qc = convert_circuit_to_instruction_set(qc, instruction_set)
    
    print("\nConverted circuit (using u3 and cx gates):")
    print(converted_qc)
    print(f"Converted gate counts: {converted_qc.count_ops()}")
    
    # Convert to a different instruction set (e.g., u, cx, rz)
    instruction_set2 = ['u', 'cx', 'rz']
    converted_qc2 = convert_circuit_to_instruction_set(qc, instruction_set2)
    
    print("\nConverted circuit (using u, cx, and rz gates):")
    print(converted_qc2)
    print(f"Converted gate counts: {converted_qc2.count_ops()}")
