"""
sensor.py - 传感器阵列模拟模块
将单像素扩展到整个图像传感器
"""

import numpy as np
from typing import List, Dict, Tuple
from tqdm import tqdm
import multiprocessing as mp
from functools import partial


class EventCameraSensor:
    """事件相机传感器阵列模拟器"""

    def __init__(self,
                 contrast_threshold: float = 0.3,
                 timestamp_resolution: float = 0.0001,
                 threshold_std: float = 0.03,
                 use_multiprocessing: bool = False,
                 num_workers: int = None):
        """
        初始化传感器

        参数:
            contrast_threshold: 对比度阈值
            timestamp_resolution: 时间戳分辨率（秒）
            threshold_std: 阈值不匹配噪声标准差
            use_multiprocessing: 是否使用多进程加速
            num_workers: 工作进程数（None=自动）
        """
        self.contrast_threshold = contrast_threshold
        self.timestamp_resolution = timestamp_resolution
        self.threshold_std = threshold_std
        self.use_multiprocessing = use_multiprocessing
        self.num_workers = num_workers or mp.cpu_count()

        # 延迟导入，避免循环依赖
        from pixel import PixelEventGenerator
        self.PixelEventGenerator = PixelEventGenerator

    def simulate(self,
                 video_frames: np.ndarray,
                 fps: float,
                 show_progress: bool = True) -> List[Dict]:
        """
        模拟整个传感器的事件生成

        参数:
            video_frames: 视频帧数组，shape=(T, H, W)
            fps: 视频帧率
            show_progress: 是否显示进度条

        返回:
            事件列表，按时间戳排序
        """
        num_frames, height, width = video_frames.shape

        # 生成时间戳
        frame_timestamps = np.arange(num_frames) / fps

        print(f"\n{'=' * 60}")
        print(f"事件相机传感器模拟")
        print(f"{'=' * 60}")
        print(f"分辨率: {width} x {height}")
        print(f"帧数: {num_frames}")
        print(f"帧率: {fps} FPS")
        print(f"时长: {frame_timestamps[-1]:.3f} 秒")
        print(f"对比度阈值: {self.contrast_threshold}")
        print(f"时间分辨率: {self.timestamp_resolution * 1000:.3f} ms")
        print(f"{'=' * 60}\n")

        # 选择处理方法
        if self.use_multiprocessing and height * width > 1000:
            print(f"使用多进程加速 (workers={self.num_workers})")
            all_events = self._simulate_multiprocess(
                video_frames, frame_timestamps, height, width, show_progress
            )
        else:
            all_events = self._simulate_sequential(
                video_frames, frame_timestamps, height, width, show_progress
            )

        # 按时间戳排序
        print("正在排序事件...")
        all_events.sort(key=lambda e: e['t'])

        # 统计信息
        self._print_statistics(all_events, frame_timestamps[-1])

        return all_events

    def _simulate_sequential(self,
                             video_frames: np.ndarray,
                             frame_timestamps: np.ndarray,
                             height: int,
                             width: int,
                             show_progress: bool) -> List[Dict]:
        """顺序处理（单进程）"""
        all_events = []
        total_pixels = height * width

        # 创建进度条
        pbar = tqdm(total=total_pixels, desc="处理像素") if show_progress else None

        for y in range(height):
            for x in range(width):
                # 提取该像素的时间序列
                pixel_values = video_frames[:, y, x]

                # 创建像素生成器
                generator = self.PixelEventGenerator(
                    contrast_threshold=self.contrast_threshold,
                    timestamp_resolution=self.timestamp_resolution,
                    threshold_std=self.threshold_std
                )

                # 生成事件
                pixel_events = generator.process(
                    pixel_values, frame_timestamps, x, y
                )

                all_events.extend(pixel_events)

                if pbar:
                    pbar.update(1)

        if pbar:
            pbar.close()

        return all_events

    def _simulate_multiprocess(self,
                               video_frames: np.ndarray,
                               frame_timestamps: np.ndarray,
                               height: int,
                               width: int,
                               show_progress: bool) -> List[Dict]:
        """多进程并行处理"""
        # 准备行数据
        row_data = [(y, video_frames[:, y, :], frame_timestamps)
                    for y in range(height)]

        # 创建处理函数
        process_func = partial(
            _process_row_worker,
            contrast_threshold=self.contrast_threshold,
            timestamp_resolution=self.timestamp_resolution,
            threshold_std=self.threshold_std
        )

        # 并行处理
        with mp.Pool(processes=self.num_workers) as pool:
            if show_progress:
                results = list(tqdm(
                    pool.imap(process_func, row_data),
                    total=height,
                    desc="处理行"
                ))
            else:
                results = pool.map(process_func, row_data)

        # 合并结果
        all_events = []
        for row_events in results:
            all_events.extend(row_events)

        return all_events

    def _print_statistics(self, events: List[Dict], duration: float):
        """打印统计信息"""
        print(f"\n{'=' * 60}")
        print(f"模拟完成 - 统计信息")
        print(f"{'=' * 60}")
        print(f"总事件数: {len(events):,}")

        if len(events) > 0:
            on_events = sum(1 for e in events if e['polarity'] == 1)
            off_events = sum(1 for e in events if e['polarity'] == -1)

            print(f"ON事件:  {on_events:,} ({on_events / len(events) * 100:.1f}%)")
            print(f"OFF事件: {off_events:,} ({off_events / len(events) * 100:.1f}%)")
            print(f"事件率: {len(events) / duration:.1f} events/sec")

            # 时间范围
            print(f"\n时间范围:")
            print(f"  首个事件: {events[0]['t']:.6f} s")
            print(f"  最后事件: {events[-1]['t']:.6f} s")

            # 空间分布统计
            x_coords = [e['x'] for e in events[:1000]]  # 采样前1000个
            y_coords = [e['y'] for e in events[:1000]]
            print(f"\n空间分布 (采样前1000个事件):")
            print(f"  X范围: {min(x_coords)} - {max(x_coords)}")
            print(f"  Y范围: {min(y_coords)} - {max(y_coords)}")
        else:
            print("警告: 未生成任何事件！")
            print("建议: 降低对比度阈值或检查输入视频")

        print(f"{'=' * 60}\n")


def _process_row_worker(row_data: Tuple,
                        contrast_threshold: float,
                        timestamp_resolution: float,
                        threshold_std: float) -> List[Dict]:
    """
    多进程工作函数：处理一行像素

    参数:
        row_data: (y, row_frames, timestamps)
        其他参数同传感器配置

    返回:
        该行的所有事件
    """
    from pixel import PixelEventGenerator

    y, row_frames, timestamps = row_data
    width = row_frames.shape[1]
    row_events = []

    for x in range(width):
        pixel_values = row_frames[:, x]

        generator = PixelEventGenerator(
            contrast_threshold=contrast_threshold,
            timestamp_resolution=timestamp_resolution,
            threshold_std=threshold_std
        )

        pixel_events = generator.process(pixel_values, timestamps, x, y)
        row_events.extend(pixel_events)

    return row_events


def test_sensor():
    """测试传感器模拟"""
    print("\n" + "=" * 60)
    print("测试传感器阵列模拟")
    print("=" * 60)

    # 创建测试视频：移动的方块
    num_frames = 60
    height, width = 128, 128
    fps = 30.0

    print(f"\n创建测试视频: {width}x{height}, {num_frames}帧, {fps}FPS")

    video_frames = np.zeros((num_frames, height, width), dtype=np.uint8)

    # 添加移动的亮方块
    for i in range(num_frames):
        # 方块从左移到右
        x_pos = int(20 + (width - 60) * i / num_frames)
        video_frames[i, 40:80, x_pos:x_pos + 20] = 200

    # 添加静止的背景
    video_frames[:, :, :] += 50  # 背景亮度

    print("视频特征: 移动的亮方块 + 静止背景")

    # 创建传感器
    sensor = EventCameraSensor(
        contrast_threshold=0.2,
        timestamp_resolution=0.001,  # 1ms
        threshold_std=0.02,
        use_multiprocessing=False  # 小图像用单进程
    )

    # 模拟
    events = sensor.simulate(video_frames, fps, show_progress=True)

    # 保存事件数据（可选）
    if len(events) > 0:
        print("\n前10个事件:")
        for i, e in enumerate(events[:10]):
            pol_str = "ON " if e['polarity'] == 1 else "OFF"
            print(f"  {i + 1}. t={e['t']:.4f}s, ({e['x']:3d},{e['y']:3d}), {pol_str}")

    return events, video_frames


if __name__ == "__main__":
    events, frames = test_sensor()