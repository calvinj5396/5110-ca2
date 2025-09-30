"""
pixel.py - 像素级事件生成模块
实现单个像素的事件检测逻辑
"""

import numpy as np
from scipy.interpolate import interp1d
from typing import List, Dict, Tuple


class PixelEventGenerator:
    """单像素事件生成器"""

    def __init__(self,
                 contrast_threshold: float = 0.3,
                 timestamp_resolution: float = 0.0001,
                 threshold_std: float = 0.03):
        """
        初始化像素事件生成器

        参数:
            contrast_threshold: 对比度阈值（C值）
            timestamp_resolution: 时间戳分辨率（秒）
            threshold_std: 阈值不匹配噪声的标准差
        """
        self.base_threshold = contrast_threshold
        self.timestamp_resolution = timestamp_resolution
        self.threshold_std = threshold_std

        # 为该像素生成独特的阈值（模拟阈值不匹配噪声）
        self.pos_threshold = self.base_threshold + np.random.normal(0, threshold_std)
        self.neg_threshold = -self.base_threshold + np.random.normal(0, threshold_std)

    def process(self,
                pixel_values: np.ndarray,
                frame_timestamps: np.ndarray,
                x: int,
                y: int) -> List[Dict]:
        """
        处理单个像素的时间序列，生成事件

        参数:
            pixel_values: 该像素在各帧的DN值 (T,)
            frame_timestamps: 对应的时间戳 (T,)
            x, y: 像素坐标

        返回:
            事件列表，每个事件包含 {x, y, t, polarity}
        """
        # 步骤1: DN值转换为对数亮度
        log_intensity = self._to_log_intensity(pixel_values)

        # 步骤2: 时间插值，生成高分辨率时间序列
        interp_timestamps, interp_intensity = self._interpolate(
            frame_timestamps, log_intensity
        )

        # 步骤3: 事件检测
        events = self._detect_events(
            interp_intensity,
            interp_timestamps,
            x, y
        )

        return events

    def _to_log_intensity(self, pixel_values: np.ndarray) -> np.ndarray:
        """
        将DN值转换为对数亮度

        参数:
            pixel_values: DN值数组

        返回:
            对数亮度数组
        """
        # 添加小常数避免log(0)
        epsilon = 1e-3
        safe_values = np.maximum(pixel_values, epsilon)

        # 归一化到[0, 1]范围（假设8位图像）
        normalized = safe_values / 255.0

        # 计算对数
        log_intensity = np.log(normalized + epsilon)

        return log_intensity

    def _interpolate(self,
                     timestamps: np.ndarray,
                     values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        对时间序列进行插值，生成高时间分辨率数据

        参数:
            timestamps: 原始时间戳
            values: 原始值

        返回:
            (插值后的时间戳, 插值后的值)
        """
        # 创建线性插值函数
        f = interp1d(timestamps, values, kind='linear',
                     fill_value='extrapolate', assume_sorted=True)

        # 生成高分辨率时间戳
        t_start = timestamps[0]
        t_end = timestamps[-1]
        high_res_timestamps = np.arange(t_start, t_end, self.timestamp_resolution)

        # 确保包含最后一个时间点
        if high_res_timestamps[-1] < t_end:
            high_res_timestamps = np.append(high_res_timestamps, t_end)

        # 插值
        interpolated_values = f(high_res_timestamps)

        return high_res_timestamps, interpolated_values

    def _detect_events(self,
                       log_intensity: np.ndarray,
                       timestamps: np.ndarray,
                       x: int,
                       y: int) -> List[Dict]:
        """
        检测事件

        参数:
            log_intensity: 对数亮度序列
            timestamps: 时间戳序列
            x, y: 像素坐标

        返回:
            事件列表
        """
        events = []

        # 初始化参考亮度
        reference_intensity = log_intensity[0]

        for i in range(1, len(log_intensity)):
            current_intensity = log_intensity[i]
            delta_log_I = current_intensity - reference_intensity

            # 检测正事件 (ON/亮度增加)
            if delta_log_I >= self.pos_threshold:
                events.append({
                    'x': x,
                    'y': y,
                    't': timestamps[i],
                    'polarity': 1  # ON事件
                })
                # 更新参考亮度（重要！）
                reference_intensity = current_intensity

            # 检测负事件 (OFF/亮度降低)
            elif delta_log_I <= self.neg_threshold:
                events.append({
                    'x': x,
                    'y': y,
                    't': timestamps[i],
                    'polarity': -1  # OFF事件
                })
                # 更新参考亮度
                reference_intensity = current_intensity

        return events


def test_pixel_generator():
    """测试函数：模拟一个像素的亮度变化"""
    print("=== 测试单像素事件生成 ===\n")

    # 创建模拟数据：30帧，模拟亮度从暗到亮再到暗
    num_frames = 30
    fps = 30
    timestamps = np.arange(num_frames) / fps  # 0到1秒

    # 模拟亮度变化：正弦波
    pixel_values = 128 + 100 * np.sin(2 * np.pi * timestamps)
    pixel_values = pixel_values.astype(np.uint8)

    print(f"输入: {num_frames}帧, FPS={fps}")
    print(f"时间范围: {timestamps[0]:.3f}s - {timestamps[-1]:.3f}s")
    print(f"亮度范围: {pixel_values.min()} - {pixel_values.max()}\n")


    # 创建生成器
    generator = PixelEventGenerator(
        contrast_threshold=0.2,
        timestamp_resolution=0.001  # 1ms
    )

    # 生成事件
    events = generator.process(pixel_values, timestamps, x=10, y=20)

    print(f"生成事件数: {len(events)}")
    print(f"\n前5个事件:")
    for i, event in enumerate(events[:5]):
        polarity_str = "ON " if event['polarity'] == 1 else "OFF"
        print(f"  {i + 1}. t={event['t']:.4f}s, ({event['x']},{event['y']}), {polarity_str}")

    # 统计
    on_events = sum(1 for e in events if e['polarity'] == 1)
    off_events = sum(1 for e in events if e['polarity'] == -1)
    print(f"\n事件统计:")
    print(f"  ON事件:  {on_events}")
    print(f"  OFF事件: {off_events}")
    print(f"  总计:    {len(events)}")


if __name__ == "__main__":
    test_pixel_generator()