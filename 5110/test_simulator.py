"""
test_simulator.py - 完整的模拟器测试脚本
生成合成测试视频并运行模拟器
"""

import numpy as np
import cv2
import os
from sensor import EventCameraSensor
from visualization import EventVisualizer


def create_synthetic_video(output_path: str,
                           num_frames: int = 120,
                           fps: float = 30.0,
                           resolution: tuple = (240, 320),
                           pattern: str = "moving_square"):
    """
    创建合成测试视频

    参数:
        output_path: 输出视频路径
        num_frames: 帧数
        fps: 帧率
        resolution: 分辨率 (height, width)
        pattern: 运动模式 ('moving_square', 'rotating', 'flashing')
    """
    height, width = resolution

    print(f"\n创建合成测试视频: {pattern}")
    print(f"  分辨率: {width}x{height}")
    print(f"  帧数: {num_frames}")
    print(f"  帧率: {fps} FPS")

    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), False)

    frames = []

    for i in range(num_frames):
        frame = np.ones((height, width), dtype=np.uint8) * 50  # 背景

        if pattern == "moving_square":
            # 从左到右移动的方块
            x_pos = int(20 + (width - 80) * i / num_frames)
            y_pos = height // 2 - 20
            frame[y_pos:y_pos + 40, x_pos:x_pos + 40] = 200

        elif pattern == "rotating":
            # 旋转的线
            angle = 360 * i / num_frames
            center = (width // 2, height // 2)
            length = min(width, height) // 3
            end_x = int(center[0] + length * np.cos(np.radians(angle)))
            end_y = int(center[1] + length * np.sin(np.radians(angle)))
            cv2.line(frame, center, (end_x, end_y), 200, 3)

        elif pattern == "flashing":
            # 闪烁的圆
            if (i // 10) % 2 == 0:  # 每10帧切换
                cv2.circle(frame, (width // 2, height // 2), 30, 200, -1)

        elif pattern == "expanding_circle":
            # 扩张的圆
            radius = int(10 + 50 * (i % 30) / 30)
            cv2.circle(frame, (width // 2, height // 2), radius, 200, 2)

        frames.append(frame)
        out.write(frame)

    out.release()
    print(f"  视频已保存: {output_path}")

    return np.array(frames), fps


def run_test(video_path: str,
             contrast_threshold: float = 0.2,
             use_multiprocessing: bool = False):
    """
    运行模拟器测试

    参数:
        video_path: 视频路径
        contrast_threshold: 对比度阈值
        use_multiprocessing: 是否使用多进程
    """
    print("\n" + "=" * 70)
    print("运行事件相机模拟器测试")
    print("=" * 70)

    # 加载视频
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        frames.append(gray)

    cap.release()
    frames = np.array(frames)

    print(f"\n视频信息:")
    print(f"  帧数: {len(frames)}")
    print(f"  分辨率: {frames.shape[2]}x{frames.shape[1]}")
    print(f"  帧率: {fps} FPS")

    # 创建传感器
    sensor = EventCameraSensor(
        contrast_threshold=contrast_threshold,
        timestamp_resolution=0.001,  # 1ms
        threshold_std=0.02,
        use_multiprocessing=use_multiprocessing
    )

    # 运行模拟
    events = sensor.simulate(frames, fps, show_progress=True)

    # 分析结果
    if len(events) > 0:
        print("\n" + "=" * 70)
        print("测试结果分析")
        print("=" * 70)

        # 时间分布
        time_span = events[-1]['t'] - events[0]['t']
        print(f"\n时间统计:")
        print(f"  时间跨度: {time_span:.3f} 秒")
        print(f"  平均事件率: {len(events) / time_span:.1f} events/sec")

        # 极性分布
        on_count = sum(1 for e in events if e['polarity'] == 1)
        off_count = len(events) - on_count
        print(f"\n极性分布:")
        print(f"  ON事件:  {on_count:6d} ({on_count / len(events) * 100:.1f}%)")
        print(f"  OFF事件: {off_count:6d} ({off_count / len(events) * 100:.1f}%)")

        # 空间分布热图
        x_coords = np.array([e['x'] for e in events])
        y_coords = np.array([e['y'] for e in events])
        print(f"\n空间统计:")
        print(f"  X范围: [{x_coords.min()}, {x_coords.max()}]")
        print(f"  Y范围: [{y_coords.min()}, {y_coords.max()}]")
        print(f"  活跃像素数: {len(set(zip(x_coords, y_coords)))}")

        # 采样显示事件
        print(f"\n事件采样 (前10个):")
        for i, e in enumerate(events[:10]):
            pol_str = "ON " if e['polarity'] == 1 else "OFF"
            print(f"  {i + 1:2d}. t={e['t']:.4f}s, ({e['x']:3d},{e['y']:3d}), {pol_str}")

        print("\n✅ 测试通过!")
    else:
        print("\n❌ 警告: 未生成任何事件!")
        print("   建议: 降低对比度阈值或检查输入视频")

    print("=" * 70 + "\n")

    return events


def test_all_patterns():
    """测试所有运动模式"""
    print("\n" + "=" * 70)
    print("综合测试：所有运动模式")
    print("=" * 70)

    patterns = [
        "moving_square",
        "rotating",
        "flashing",
        "expanding_circle"
    ]

    test_dir = "test_videos"
    os.makedirs(test_dir, exist_ok=True)

    results = {}

    for pattern in patterns:
        print(f"\n{'=' * 70}")
        print(f"测试模式: {pattern}")
        print(f"{'=' * 70}")

        # 创建测试视频
        video_path = os.path.join(test_dir, f"{pattern}.mp4")
        frames, fps = create_synthetic_video(
            video_path,
            num_frames=60,
            fps=30.0,
            resolution=(180, 240),
            pattern=pattern
        )

        # 运行测试
        events = run_test(video_path, contrast_threshold=0.15)

        results[pattern] = {
            'num_events': len(events),
            'video_path': video_path
        }

    # 总结
    print("\n" + "=" * 70)
    print("测试总结")
    print("=" * 70)
    for pattern, result in results.items():
        print(f"{pattern:20s}: {result['num_events']:6d} events")
    print("=" * 70 + "\n")


def quick_test():
    """快速测试：单个简单场景"""
    print("\n" + "=" * 70)
    print("快速测试")
    print("=" * 70)

    test_dir = "test_videos"
    os.makedirs(test_dir, exist_ok=True)

    video_path = os.path.join(test_dir, "quick_test.mp4")

    # 创建简单的测试视频
    frames, fps = create_synthetic_video(
        video_path,
        num_frames=60,
        fps=30.0,
        resolution=(120, 160),
        pattern="moving_square"
    )

    # 运行测试
    events = run_test(video_path, contrast_threshold=0.2)

    return events


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "all":
        # 运行所有模式测试
        test_all_patterns()
    else:
        # 快速测试
        quick_test()

        print("\n提示: 运行 'python test_simulator.py all' 测试所有模式")