"""
visualization.py - 事件可视化模块
将事件数据叠加到原始视频上，生成演示视频
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple
from tqdm import tqdm


class EventVisualizer:
    """事件可视化器"""

    def __init__(self,
                 width: int,
                 height: int,
                 accumulation_time: float = 0.033,
                 style: str = "red_blue",
                 decay: bool = True,
                 decay_rate: float = 0.5):
        """
        初始化可视化器

        参数:
            width, height: 图像尺寸
            accumulation_time: 事件累积时间窗口（秒）
            style: 可视化风格
                - "red_blue": ON=蓝色, OFF=红色
                - "green_red": ON=绿色, OFF=红色
                - "white": 所有事件为白色
                - "heatmap": 热图模式
            decay: 是否使用衰减效果
            decay_rate: 衰减率（0-1）
        """
        self.width = width
        self.height = height
        self.accumulation_time = accumulation_time
        self.style = style
        self.decay = decay
        self.decay_rate = decay_rate

        # 颜色方案
        self.color_schemes = {
            "red_blue": {
                1: (255, 0, 0),  # ON: 蓝色 (BGR)
                -1: (0, 0, 255)  # OFF: 红色
            },
            "green_red": {
                1: (0, 255, 0),  # ON: 绿色
                -1: (0, 0, 255)  # OFF: 红色
            },
            "white": {
                1: (255, 255, 255),  # ON: 白色
                -1: (255, 255, 255)  # OFF: 白色
            }
        }

    def create_video(self,
                     events: List[Dict],
                     original_frames: np.ndarray,
                     fps: float,
                     output_path: str,
                     show_progress: bool = True) -> None:
        """
        创建事件可视化视频

        参数:
            events: 事件列表
            original_frames: 原始视频帧 (T, H, W)
            fps: 输出视频帧率
            output_path: 输出视频路径
            show_progress: 是否显示进度条
        """
        print(f"\n{'=' * 70}")
        print(f"创建事件可视化视频")
        print(f"{'=' * 70}")
        print(f"输出路径: {output_path}")
        print(f"分辨率: {self.width}x{self.height}")
        print(f"帧率: {fps} FPS")
        print(f"累积时间: {self.accumulation_time * 1000:.1f} ms")
        print(f"可视化风格: {self.style}")
        print(f"事件总数: {len(events)}")
        print(f"{'=' * 70}\n")

        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps,
                              (self.width, self.height), True)

        num_frames = len(original_frames)
        frame_duration = 1.0 / fps

        # 事件索引（加速查找）
        event_idx = 0

        # 创建累积图层（用于衰减效果）
        event_layer = np.zeros((self.height, self.width, 3), dtype=np.float32)

        # 进度条
        pbar = tqdm(total=num_frames, desc="渲染帧") if show_progress else None

        for frame_idx in range(num_frames):
            # 当前帧的时间窗口
            t_start = frame_idx * frame_duration
            t_end = t_start + self.accumulation_time

            # 获取原始帧
            frame = original_frames[frame_idx]

            # 转换为彩色（如果是灰度图）
            if len(frame.shape) == 2:
                frame_color = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            else:
                frame_color = frame.copy()

            # 衰减之前的事件
            if self.decay:
                event_layer *= self.decay_rate
            else:
                event_layer.fill(0)

            # 收集当前窗口的事件
            window_events = []
            temp_idx = event_idx

            while temp_idx < len(events) and events[temp_idx]['t'] < t_end:
                if events[temp_idx]['t'] >= t_start:
                    window_events.append(events[temp_idx])
                temp_idx += 1

            # 更新事件索引
            while event_idx < len(events) and events[event_idx]['t'] < t_start:
                event_idx += 1

            # 渲染事件
            if self.style == "heatmap":
                event_layer = self._render_heatmap(window_events, event_layer)
            else:
                event_layer = self._render_overlay(window_events, event_layer)

            # 合成最终图像
            vis_frame = self._blend_frames(frame_color, event_layer)

            # 添加信息文本
            vis_frame = self._add_info_text(
                vis_frame, frame_idx, num_frames,
                len(window_events), t_start
            )

            # 写入视频
            out.write(vis_frame)

            if pbar:
                pbar.update(1)

        if pbar:
            pbar.close()

        out.release()

        print(f"\n✅ 视频生成完成: {output_path}")
        print(f"{'=' * 70}\n")

    def _render_overlay(self,
                        events: List[Dict],
                        event_layer: np.ndarray) -> np.ndarray:
        """渲染叠加模式的事件"""
        colors = self.color_schemes[self.style]

        for event in events:
            x, y = event['x'], event['y']
            polarity = event['polarity']

            # 边界检查
            if 0 <= x < self.width and 0 <= y < self.height:
                color = colors[polarity]
                # 累加颜色（而不是直接覆盖）
                event_layer[y, x] = np.minimum(
                    event_layer[y, x] + np.array(color, dtype=np.float32),
                    255.0
                )

        return event_layer

    def _render_heatmap(self,
                        events: List[Dict],
                        event_layer: np.ndarray) -> np.ndarray:
        """渲染热图模式"""
        # 创建事件密度图
        density = np.zeros((self.height, self.width), dtype=np.float32)

        for event in events:
            x, y = event['x'], event['y']
            if 0 <= x < self.width and 0 <= y < self.height:
                density[y, x] += 1.0

        # 添加到累积层
        event_layer[:, :, 0] += density

        # 归一化并应用colormap
        if density.max() > 0:
            normalized = (density / density.max() * 255).astype(np.uint8)
            colored = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
            event_layer = colored.astype(np.float32)

        return event_layer

    def _blend_frames(self,
                      original: np.ndarray,
                      event_layer: np.ndarray,
                      alpha: float = 0.6) -> np.ndarray:
        """
        混合原始帧和事件层

        参数:
            original: 原始帧
            event_layer: 事件层
            alpha: 事件层透明度
        """
        # 将原始帧略微调暗，让事件更明显
        darkened = (original * 0.7).astype(np.uint8)

        # 创建掩码（有事件的地方）
        mask = (event_layer.sum(axis=2) > 0).astype(np.float32)
        mask = np.stack([mask] * 3, axis=2)

        # 混合
        event_uint8 = event_layer.astype(np.uint8)
        blended = cv2.addWeighted(darkened, 1.0, event_uint8, alpha, 0)

        return blended

    def _add_info_text(self,
                       frame: np.ndarray,
                       frame_idx: int,
                       total_frames: int,
                       num_events: int,
                       timestamp: float) -> np.ndarray:
        """添加信息文本到帧上"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        color = (255, 255, 255)
        bg_color = (0, 0, 0)

        # 准备文本
        texts = [
            f"Frame: {frame_idx + 1}/{total_frames}",
            f"Time: {timestamp:.3f}s",
            f"Events: {num_events}"
        ]

        y_offset = 20
        for text in texts:
            # 获取文本大小
            (text_width, text_height), _ = cv2.getTextSize(
                text, font, font_scale, thickness
            )

            # 绘制黑色背景
            cv2.rectangle(frame,
                          (5, y_offset - text_height - 2),
                          (15 + text_width, y_offset + 2),
                          bg_color, -1)

            # 绘制文本
            cv2.putText(frame, text, (10, y_offset),
                        font, font_scale, color, thickness)

            y_offset += 25

        return frame

    def create_comparison_video(self,
                                events: List[Dict],
                                original_frames: np.ndarray,
                                fps: float,
                                output_path: str) -> None:
        """
        创建对比视频（原始 | 事件 | 叠加）

        参数:
            events: 事件列表
            original_frames: 原始视频帧
            fps: 帧率
            output_path: 输出路径
        """
        print(f"\n创建三栏对比视频...")

        # 创建三个可视化器
        viz_events_only = EventVisualizer(
            self.width, self.height,
            accumulation_time=self.accumulation_time,
            style=self.style,
            decay=False
        )

        viz_overlay = EventVisualizer(
            self.width, self.height,
            accumulation_time=self.accumulation_time,
            style=self.style,
            decay=True
        )

        # 输出宽度 = 3 * 原宽度
        out_width = self.width * 3
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps,
                              (out_width, self.height), True)

        num_frames = len(original_frames)
        frame_duration = 1.0 / fps
        event_idx = 0

        for frame_idx in tqdm(range(num_frames), desc="渲染对比视频"):
            t_start = frame_idx * frame_duration
            t_end = t_start + self.accumulation_time

            # 原始帧
            frame = original_frames[frame_idx]
            if len(frame.shape) == 2:
                frame_color = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            else:
                frame_color = frame.copy()

            # 收集事件
            window_events = []
            temp_idx = event_idx
            while temp_idx < len(events) and events[temp_idx]['t'] < t_end:
                if events[temp_idx]['t'] >= t_start:
                    window_events.append(events[temp_idx])
                temp_idx += 1

            while event_idx < len(events) and events[event_idx]['t'] < t_start:
                event_idx += 1

            # 创建事件可视化（纯黑背景）
            event_only = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            event_layer = np.zeros((self.height, self.width, 3), dtype=np.float32)
            event_layer = viz_events_only._render_overlay(window_events, event_layer)
            event_only = event_layer.astype(np.uint8)

            # 创建叠加可视化
            event_layer2 = np.zeros((self.height, self.width, 3), dtype=np.float32)
            event_layer2 = viz_overlay._render_overlay(window_events, event_layer2)
            overlay = viz_overlay._blend_frames(frame_color, event_layer2)

            # 添加标签
            cv2.putText(frame_color, "Original", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(event_only, "Events Only", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(overlay, "Overlay", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # 水平拼接
            combined = np.hstack([frame_color, event_only, overlay])

            out.write(combined)

        out.release()
        print(f"✅ 对比视频生成完成: {output_path}\n")


def create_event_gif(events: List[Dict],
                     width: int,
                     height: int,
                     duration: float,
                     output_path: str,
                     num_frames: int = 30):
    """
    创建事件动画GIF（用于快速预览）

    参数:
        events: 事件列表
        width, height: 尺寸
        duration: 总时长（秒）
        output_path: 输出GIF路径
        num_frames: GIF帧数
    """
    print(f"\n创建事件预览GIF...")

    frames = []
    time_step = duration / num_frames

    for i in range(num_frames):
        t_start = i * time_step
        t_end = t_start + time_step * 2  # 累积2个时间步

        # 创建黑色背景
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # 收集时间窗口内的事件
        window_events = [e for e in events
                         if t_start <= e['t'] < t_end]

        # 绘制事件
        for event in window_events:
            x, y = event['x'], event['y']
            if 0 <= x < width and 0 <= y < height:
                color = (0, 255, 0) if event['polarity'] == 1 else (0, 0, 255)
                cv2.circle(frame, (x, y), 1, color, -1)

        frames.append(frame)

    # 使用OpenCV保存（如果需要真正的GIF，建议用PIL）
    print(f"预览帧数: {len(frames)}")
    print(f"(注意: 需要额外的库来生成GIF，这里生成视频代替)")


if __name__ == "__main__":
    print("可视化模块已加载")
    print("使用示例:")
    print("  from visualization import EventVisualizer")
    print("  viz = EventVisualizer(width, height)")
    print("  viz.create_video(events, frames, fps, 'output.mp4')")