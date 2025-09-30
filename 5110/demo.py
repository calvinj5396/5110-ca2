"""
demo.py - 完整演示脚本
展示事件相机模拟器的所有功能
"""

import numpy as np
import cv2
import os
from sensor import EventCameraSensor
from visualization import EventVisualizer


def create_demo_video(output_dir="demo_output"):
    """创建完整的演示材料"""

    print("\n" + "=" * 70)
    print("事件相机模拟器 - 完整演示")
    print("=" * 70)

    os.makedirs(output_dir, exist_ok=True)

    # ===== 场景3: 多个运动物体 =====
    print("\n场景3: 多个运动物体")
    print("-" * 70)

    scene3_frames = create_multiple_objects(
        num_frames=120,
        resolution=(240, 320)
    )

    scene3_input = os.path.join(output_dir, "scene3_input.mp4")
    save_video(scene3_frames, fps, scene3_input)

    # 使用更低的阈值捕捉更多事件
    sensor_sensitive = EventCameraSensor(
        contrast_threshold=0.15,
        timestamp_resolution=0.001,
        threshold_std=0.02
    )
    events3 = sensor_sensitive.simulate(scene3_frames, fps)

    viz3 = EventVisualizer(320, 240, style='red_blue', accumulation_time=0.05, decay=True)
    scene3_viz = os.path.join(output_dir, "scene3_visualization.mp4")
    viz3.create_video(events3, scene3_frames, fps, scene3_viz, show_progress=False)

    # ===== 生成统计报告 =====
    print("\n生成统计报告...")
    report_path = os.path.join(output_dir, "demo_report.txt")

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("事件相机模拟器 - 演示报告\n")
        f.write("=" * 70 + "\n\n")

        f.write("场景1: 移动的方块\n")
        f.write(f"  事件总数: {len(events1):,}\n")
        f.write(f"  ON事件: {sum(1 for e in events1 if e['polarity'] == 1):,}\n")
        f.write(f"  OFF事件: {sum(1 for e in events1 if e['polarity'] == -1):,}\n")
        f.write(f"  输出视频: scene1_*.mp4\n\n")

        f.write("场景2: 旋转的线\n")
        f.write(f"  事件总数: {len(events2):,}\n")
        f.write(f"  ON事件: {sum(1 for e in events2 if e['polarity'] == 1):,}\n")
        f.write(f"  OFF事件: {sum(1 for e in events2 if e['polarity'] == -1):,}\n")
        f.write(f"  输出视频: scene2_visualization.mp4\n\n")

        f.write("场景3: 多个运动物体\n")
        f.write(f"  事件总数: {len(events3):,}\n")
        f.write(f"  ON事件: {sum(1 for e in events3 if e['polarity'] == 1):,}\n")
        f.write(f"  OFF事件: {sum(1 for e in events3 if e['polarity'] == -1):,}\n")
        f.write(f"  输出视频: scene3_visualization.mp4\n\n")

        f.write("可视化风格演示:\n")
        f.write("  - red_blue: ON=蓝色, OFF=红色\n")
        f.write("  - green_red: ON=绿色, OFF=红色\n")
        f.write("  - heatmap: 事件密度热图\n\n")

        f.write("对比视频:\n")
        f.write("  scene1_comparison.mp4 展示了三栏对比效果\n")

    print(f"报告已保存: {report_path}")

    # 打印总结
    print("\n" + "=" * 70)
    print("演示完成！")
    print("=" * 70)
    print(f"输出目录: {output_dir}/")
    print(f"\n生成的文件:")
    for filename in sorted(os.listdir(output_dir)):
        filepath = os.path.join(output_dir, filename)
        size_mb = os.path.getsize(filepath) / 1024 / 1024
        print(f"  {filename:40s} ({size_mb:6.2f} MB)")
    print("=" * 70 + "\n")


def create_moving_square(num_frames, fps, resolution):
    """创建移动方块场景"""
    height, width = resolution
    frames = []

    for i in range(num_frames):
        frame = np.ones((height, width), dtype=np.uint8) * 50

        # 方块从左到右移动
        x_pos = int(20 + (width - 80) * i / num_frames)
        y_pos = height // 2 - 20

        # 绘制方块
        frame[y_pos:y_pos + 40, x_pos:x_pos + 40] = 200

        frames.append(frame)

    return np.array(frames), fps


def create_rotating_line(num_frames, resolution):
    """创建旋转线场景"""
    height, width = resolution
    frames = []

    for i in range(num_frames):
        frame = np.ones((height, width), dtype=np.uint8) * 50

        # 旋转的线
        angle = 360 * i / num_frames
        center = (width // 2, height // 2)
        length = min(width, height) // 3

        end_x = int(center[0] + length * np.cos(np.radians(angle)))
        end_y = int(center[1] + length * np.sin(np.radians(angle)))

        cv2.line(frame, center, (end_x, end_y), 220, 3)

        frames.append(frame)

    return np.array(frames)


def create_multiple_objects(num_frames, resolution):
    """创建多个运动物体场景"""
    height, width = resolution
    frames = []

    for i in range(num_frames):
        frame = np.ones((height, width), dtype=np.uint8) * 50

        # 物体1: 水平移动的圆
        x1 = int(30 + (width - 60) * i / num_frames)
        y1 = height // 3
        cv2.circle(frame, (x1, y1), 15, 200, -1)

        # 物体2: 垂直移动的方块
        x2 = width // 2
        y2 = int(30 + (height - 60) * i / num_frames)
        frame[y2 - 15:y2 + 15, x2 - 15:x2 + 15] = 180

        # 物体3: 对角线移动的三角形
        x3 = int(30 + (width - 60) * i / num_frames)
        y3 = int(30 + (height - 60) * i / num_frames)
        pts = np.array([
            [x3, y3 - 20],
            [x3 - 15, y3 + 10],
            [x3 + 15, y3 + 10]
        ], np.int32)
        cv2.fillPoly(frame, [pts], 220)

        frames.append(frame)

    return np.array(frames)


def save_video(frames, fps, output_path):
    """保存视频"""
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), False)

    for frame in frames:
        out.write(frame)

    out.release()


def create_presentation_slides():
    """创建演示幻灯片的素材"""
    print("\n创建演示幻灯片素材...")

    slides_dir = "demo_output/slides"
    os.makedirs(slides_dir, exist_ok=True)

    # 创建一些关键帧截图
    print("  提示: 使用生成的视频制作幻灯片")
    print("  建议幻灯片内容:")
    print("    1. 项目介绍 - 什么是事件相机")
    print("    2. 算法原理 - 对数亮度变化检测")
    print("    3. 系统架构 - 像素→阵列→可视化")
    print("    4. 演示场景1 - 移动方块")
    print("    5. 演示场景2 - 旋转线")
    print("    6. 演示场景3 - 多物体")
    print("    7. 可视化对比 - 不同风格")
    print("    8. 性能分析 - 事件统计")
    print("    9. 未来改进 - 噪声模型等")
    print("   10. Q&A")


def quick_demo():
    """快速演示（小尺寸，快速生成）"""
    print("\n快速演示模式 (用于测试)")
    print("=" * 70)

    output_dir = "quick_demo"
    os.makedirs(output_dir, exist_ok=True)

    # 小分辨率，少帧数
    frames, fps = create_moving_square(
        num_frames=30,
        fps=15.0,
        resolution=(120, 160)
    )

    # 保存输入
    input_path = os.path.join(output_dir, "input.mp4")
    save_video(frames, fps, input_path)

    # 模拟
    sensor = EventCameraSensor(
        contrast_threshold=0.2,
        timestamp_resolution=0.001
    )
    events = sensor.simulate(frames, fps)

    # 可视化
    viz = EventVisualizer(160, 120, style='red_blue')
    output_path = os.path.join(output_dir, "output.mp4")
    viz.create_video(events, frames, fps, output_path)

    print(f"\n✅ 快速演示完成！")
    print(f"   输入: {input_path}")
    print(f"   输出: {output_path}")
    print(f"   事件数: {len(events)}\n")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "quick":
            quick_demo()
        elif sys.argv[1] == "slides":
            create_presentation_slides()
        else:
            print("用法:")
            print("  python demo.py        - 完整演示")
            print("  python demo.py quick  - 快速演示")
            print("  python demo.py slides - 幻灯片建议")
    else:
        create_demo_video()
1: 移动的方块 == == =
print("\n场景1: 移动的方块")
print("-" * 70)

scene1_frames, fps = create_moving_square(
    num_frames=90,
    fps=30.0,
    resolution=(240, 320)
)

# 保存输入视频
scene1_input = os.path.join(output_dir, "scene1_input.mp4")
save_video(scene1_frames, fps, scene1_input)

# 模拟事件
sensor = EventCameraSensor(
    contrast_threshold=0.2,
    timestamp_resolution=0.001,
    threshold_std=0.02
)
events1 = sensor.simulate(scene1_frames, fps)

# 可视化 - 多种风格
styles = ['red_blue', 'green_red', 'heatmap']

for style in styles:
    print(f"\n  生成 {style} 风格可视化...")
    viz = EventVisualizer(
        width=320,
        height=240,
        accumulation_time=0.033,
        style=style,
        decay=True
    )

    output_path = os.path.join(output_dir, f"scene1_{style}.mp4")
    viz.create_video(events1, scene1_frames, fps, output_path, show_progress=False)

# 对比视频
print("\n  生成三栏对比视频...")
viz_comp = EventVisualizer(320, 240, style='red_blue')
comp_path = os.path.join(output_dir, "scene1_comparison.mp4")
viz_comp.create_comparison_video(events1, scene1_frames, fps, comp_path)

# ===== 场景2: 旋转的线 =====
print("\n场景2: 旋转的线")
print("-" * 70)

scene2_frames = create_rotating_line(
    num_frames=120,
    resolution=(240, 320)
)

scene2_input = os.path.join(output_dir, "scene2_input.mp4")
save_video(scene2_frames, fps, scene2_input)

events2 = sensor.simulate(scene2_frames, fps)

viz2 = EventVisualizer(320, 240, style='green_red', accumulation_time=0.02)
scene2_viz = os.path.join(output_dir, "scene2_visualization.mp4")
viz2.create_video(events2, scene2_frames, fps, scene2_viz, show_progress=False)

# ===== 场景