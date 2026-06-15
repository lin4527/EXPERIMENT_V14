#!/usr/bin/env python3
"""
医学图像处理综合实验 14 — 脊椎椎体逐节分割系统
主程序入口

用法:
    python main.py              # 启动 GUI
    python main.py --test       # 运行所有模块测试
    python main.py --module 1   # 测试指定模块
"""
import sys
import os


def setup_path():
    """确保项目根目录在 sys.path 中"""
    root = os.path.dirname(os.path.abspath(__file__))
    if root not in sys.path:
        sys.path.insert(0, root)


def run_gui():
    """启动 GUI 主界面"""
    from PyQt5.QtWidgets import QApplication
    from module6_gui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    # 全局字体
    font = app.font()
    font.setPointSize(9)
    app.setFont(font)

    window = MainWindow()
    window.show()

    print("=" * 60)
    print("医学图像处理综合实验 14 — 脊椎椎体逐节分割系统")
    print("=" * 60)
    print("[OK] GUI 已启动")
    print("左侧面板: 文件导入 | 预处理 | 传统分割 | 深度学习 | 评估")
    print("右侧: MPR三视图 + 原图/结果双画布")
    print("点击左侧原图可选取区域生长种子点")
    print("=" * 60)

    sys.exit(app.exec_())


def run_all_tests():
    """运行所有模块独立测试"""
    print("\n" + "=" * 60)
    print(">>> 运行所有模块测试")
    print("=" * 60)

    tests = [
        ("模块1: 数据读取与可视化", "module1_io_visualization.test_module1"),
        ("模块2: 图像预处理", "module2_preprocessing.test_module2"),
        ("模块3: 传统分割", "module3_traditional_seg.test_module3"),
    ]

    passed = 0
    failed = 0

    for name, module_path in tests:
        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")
        try:
            mod = __import__(module_path, fromlist=['test'])
            if hasattr(mod, 'test_module1'):
                mod.test_module1()
            elif hasattr(mod, 'test_module2'):
                mod.test_module2()
            elif hasattr(mod, 'test_module3'):
                mod.test_module3()
            print(f"  [OK] {name} PASSED")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*60}")
    print(f"  测试结果: {passed} passed, {failed} failed")
    print(f"{'='*60}")

    return failed == 0


def run_module_test(module_num):
    """运行指定模块测试"""
    test_map = {
        '1': ('module1_io_visualization.test_module1', 'test_module1'),
        '2': ('module2_preprocessing.test_module2', 'test_module2'),
        '3': ('module3_traditional_seg.test_module3', 'test_module3'),
    }

    if module_num not in test_map:
        print(f"未知模块: {module_num}")
        print(f"可选: {list(test_map.keys())}")
        return False

    mod_path, func_name = test_map[module_num]
    mod = __import__(mod_path, fromlist=[func_name])
    test_func = getattr(mod, func_name)
    return test_func()


def main():
    setup_path()

    if '--test' in sys.argv:
        run_all_tests()
    elif '--module' in sys.argv:
        try:
            idx = sys.argv.index('--module')
            module_num = sys.argv[idx + 1]
            run_module_test(module_num)
        except (IndexError, ValueError):
            print("用法: python main.py --module <1|2|3>")
    else:
        run_gui()


if __name__ == "__main__":
    main()
