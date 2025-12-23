"""
Janus Circuit 命令行工具

从 JSON 文件加载电路并执行测试
"""
import argparse
import sys

from .io import load_circuit


def cmd_info(args):
    """显示电路信息"""
    circuit = load_circuit(filepath=args.file)
    
    print(f"电路名称: {circuit.name or '未命名'}")
    print(f"量子比特数: {circuit.n_qubits}")
    print(f"经典比特数: {circuit.n_clbits}")
    print(f"门数量: {circuit.n_gates}")
    print(f"电路深度: {circuit.depth}")
    print(f"两比特门数量: {circuit.num_two_qubit_gates}")
    
    if args.verbose:
        print(f"\n指令列表:")
        for i, inst in enumerate(circuit.instructions):
            clbits_str = f", clbits={inst.clbits}" if inst.clbits else ""
            params_str = f", params={inst.params}" if inst.params else ""
            print(f"  [{i}] {inst.name}: qubits={inst.qubits}{clbits_str}{params_str}")


def cmd_draw(args):
    """绘制电路"""
    circuit = load_circuit(filepath=args.file)
    
    if args.output:
        circuit.draw(output='png', filename=args.output, dpi=args.dpi)
    else:
        print(circuit.draw(output='text'))


def cmd_test(args):
    """测试电路功能"""
    circuit = load_circuit(filepath=args.file)
    
    print(f"测试电路: {circuit.name or args.file}")
    print("=" * 50)
    
    print(f"✓ 加载成功")
    print(f"  - 量子比特: {circuit.n_qubits}")
    print(f"  - 经典比特: {circuit.n_clbits}")
    print(f"  - 门数量: {circuit.n_gates}")
    print(f"  - 深度: {circuit.depth}")
    
    # 测试复制
    circuit_copy = circuit.copy()
    assert circuit_copy.n_gates == circuit.n_gates
    print(f"✓ 复制测试通过")
    
    # 测试分层
    layers = circuit.layers
    print(f"✓ 分层计算通过 ({len(layers)} 层)")
    
    # 检查测量门
    measure_count = sum(1 for inst in circuit.instructions if inst.name == 'measure')
    if measure_count > 0:
        print(f"✓ 包含 {measure_count} 个测量门")
    
    print("\n所有测试通过!")


def main():
    parser = argparse.ArgumentParser(
        description='Janus Circuit 命令行工具',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # info 命令
    info_parser = subparsers.add_parser('info', help='显示电路信息')
    info_parser.add_argument('file', help='电路 JSON 文件')
    info_parser.add_argument('-v', '--verbose', action='store_true', help='显示详细信息')
    info_parser.set_defaults(func=cmd_info)
    
    # draw 命令
    draw_parser = subparsers.add_parser('draw', help='绘制电路')
    draw_parser.add_argument('file', help='电路 JSON 文件')
    draw_parser.add_argument('-o', '--output', help='输出图片文件')
    draw_parser.add_argument('--dpi', type=int, default=150, help='图片 DPI')
    draw_parser.set_defaults(func=cmd_draw)
    
    # test 命令
    test_parser = subparsers.add_parser('test', help='测试电路功能')
    test_parser.add_argument('file', help='电路 JSON 文件')
    test_parser.set_defaults(func=cmd_test)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    args.func(args)


if __name__ == '__main__':
    main()
