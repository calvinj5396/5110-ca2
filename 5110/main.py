"""
main.py - 事件相机模拟器主程序
"""

import numpy as np
import cv2
import argparse
import os
from datetime import datetime
from sensor import EventCameraSensor
from visualization import EventVisualizer


def load_video(video_path: str, max_frames: int = None):
    """
    加载视频文件

    参数:
        video_path: 视频文件路径
        max_frames: 最大帧数限制（None=全部加载）

    返回:
        (frames, fps, original_size)
    """
    print(f"\n正在加载视频: {video_path}")

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")

    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"视频信息:")
    print(f"  分辨率: {width}x{height}")
    print(f"  帧率: {fps} FPS")
    print(f"  总帧数: {total_frames}")
    print(f"  时长: {total_frames / fps:.2f} 秒")

    # 读取帧
    frames = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 转换为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray)
        frame_count += 1

        # 检查是否达到最大帧数
        if max_frames and frame_count >= max_frames:
            print(f"  (已限制为前 {max_frames} 帧)")
            break

    cap.release()

    frames = np.array(frames)
    print(f"成功加载 {len(frames)} 帧")

    return frames, fps, (width, height)


def save_events(events, output_path: str):
    """
    保存事件数据

    参数:
        events: 事件列表
        output_path: 输出文件路径（.txt或.npy）
    """
    print(f"\n正在保存事件数据: {output_path}")

    ext = os.path.splitext(output_path)[1]

    if ext == '.txt':
        # 保存为文本格式: t x y polarity
        with open(output_path, 'w') as f:
            f.write("# t(s) x y polarity\n")
            for e in events:
                f.write(f"{e['t']:.6f} {e['x']} {e['y']} {e['polarity']}\n")

    elif ext == '.npy':
        # 保存为numpy二进制格式（更紧凑）
        event_array = np.array([
            (e['t'], e['x'], e['y'], e['polarity'])
            for e in events
        ], dtype=[('t', 'f8'), ('x', 'i4'), ('y', 'i4'), ('polarity', 'i4')])
        np.save(output_path, event_array)

    else:
        raise ValueError(f"不支持的文件格式: {ext}，请使用 .txt 或 .npy")

    file_size = os.path.getsize(output_path) / 1024 / 1024  # MB
    print(f"保存成功! 文件大小: {file_size:.2f} MB")


def create_output_directory(base_dir: str = "output") -> str:
    """创建输出目录"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, f"sim_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="事件相机模拟器",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # 输入输出参数
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='输入视频文件路径')
    parser.add_argument('--output', '-o', type=str, default='output',
                        help='输出目录')
    parser.add_argument('--max-frames', type=int, default=None,
                        help='最大处理帧数（用于测试）')

    # 模拟器参数
    parser.add_argument('--threshold', '-C', type=float, default=0.3,
                        help='对比度阈值')
    parser.add_argument('--time-resolution', type=float, default=0.0001,
                        help='时间戳分辨率（秒）')
    parser.add_argument('--threshold-std', type=float, default=0.03,
                        help='阈值不匹配噪声标准差')

    # 性能参数
    parser.add_argument('--multiprocessing', action='store_true',
                        help='使用多进程加速')
    parser.add_argument('--workers', type=int, default=None,
                        help='工作进程数（默认=CPU核心数）')

    # 输出格式
    parser.add_argument('--format', choices=['txt', 'npy'], default='txt',
                        help='事件数据保存格式')

    # 可视化参数
    parser.add_argument('--visualize', action='store_true',
                        help='生成可视化视频')
    parser.add_argument('--viz-style', choices=['red_blue', 'green_red', 'white', 'heatmap'],
                        default='red_blue', help='可视化风格')
    parser.add_argument('--accumulation-time', type=float, default=0.033,
                        help='事件累积时间（秒），默认33ms')
    parser.add_argument('--comparison', action='store_true',
                        help='生成三栏对比视频')
    parser.add_argument('--output-fps', type=float, default=None,
                        help='输出视频帧率（默认=输入帧率）')

    args = parser.parse_args()

    # 打印配置
    print("\n" + "=" * 60)
    print("事件相机模拟器")
    print("=" * 60)
    print("\n配置参数:")
    print(f"  输入视频: {args.input}")
    print(f"  输出目录: {args.output}")
    print(f"  对比度阈值: {args.threshold}")
    print(f"  时间分辨率: {args.time_resolution * 1000} ms")
    print(f"  阈值噪声: ±{args.threshold_std}")
    print(f"  多进程: {'是' if args.multiprocessing else '否'}")
    if args.visualize:
        print(f"  可视化: 是 ({args.viz_style})")
        print(f"  累积时间: {args.accumulation_time * 1000} ms")

    try:
        # 步骤1: 加载视频
        frames, fps, size = load_video(args.input, args.max_frames)

        # 步骤2: 创建传感器
        sensor = EventCameraSensor(
            contrast_threshold=args.threshold,
            timestamp_resolution=args.time_resolution,
            threshold_std=args.threshold_std,
            use_multiprocessing=args.multiprocessing,
            num_workers=args.workers
        )

        # 步骤3: 模拟事件生成
        events = sensor.simulate(frames, fps, show_progress=True)

        # 步骤4: 保存结果
        output_dir = create_output_directory(args.output)
        print(f"\n输出目录: {output_dir}")

        # 保存事件数据
        event_file = os.path.join(output_dir, f"events.{args.format}")
        save_events(events, event_file)

        # 保存配置信息
        config_file = os.path.join(output_dir, "config.txt")
        with open(config_file, 'w') as f:
            f.write("事件相机模拟器配置\n")
            f.write("=" * 40 + "\n")
            f.write(f"输入视频: {args.input}\n")
            f.write(f"分辨率: {size[0]}x{size[1]}\n")
            f.write(f"帧率: {fps} FPS\n")
            f.write(f"帧数: {len(frames)}\n")
            f.write(f"对比度阈值: {args.threshold}\n")
            f.write(f"时间分辨率: {args.time_resolution} s\n")
            f.write(f"阈值噪声标准差: {args.threshold_std}\n")
            f.write(f"\n生成事件总数: {len(events)}\n")

        print(f"配置已保存: {config_file}")

        # 步骤5: 生成可视化（如果需要）
        if args.visualize and len(events) > 0:
            print(f"\n{'=' * 60}")
            print("生成可视化视频")
            print(f"{'=' * 60}")

            output_fps = args.output_fps if args.output_fps else fps

            visualizer = EventVisualizer(
                width=size[0],
                height=size[1],
                accumulation_time=args.accumulation_time,
                style=args.viz_style,
                decay=True
            )

            if args.comparison:
                # 三栏对比视频
                viz_file = os.path.join(output_dir, "comparison.mp4")
                visualizer.create_comparison_video(
                    events, frames, output_fps, viz_file
                )
            else:
                # 标准叠加视频
                viz_file = os.path.join(output_dir, "visualization.mp4")
                visualizer.create_video(
                    events, frames, output_fps, viz_file, show_progress=True
                )

            print(f"可视化视频已保存: {viz_file}")
        elif args.visualize and len(events) == 0:
            print("\n警告: 无法生成可视化，因为没有事件数据")

        print("\n" + "=" * 60)
        print("模拟完成！")
        print("=" * 60)

    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())