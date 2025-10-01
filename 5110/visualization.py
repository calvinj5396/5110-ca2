"""
visualization.py - Event visualization module
Overlay event data on the original video to produce demo videos
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple
from tqdm import tqdm


class EventVisualizer:
    """Event visualizer"""

    def __init__(self,
                 width: int,
                 height: int,
                 accumulation_time: float = 0.033,
                 style: str = "red_blue",
                 decay: bool = True,
                 decay_rate: float = 0.5):
        """
        Initialize the visualizer

        Args:
            width, height: Image size
            accumulation_time: Event accumulation window (seconds)
            style: Visualization style
                - "red_blue": ON=Blue, OFF=Red
                - "green_red": ON=Green, OFF=Red
                - "white": All events are white
                - "heatmap": Heatmap mode
            decay: Whether to use decay effect
            decay_rate: Decay rate (0-1)
        """
        self.width = width
        self.height = height
        self.accumulation_time = accumulation_time
        self.style = style
        self.decay = decay
        self.decay_rate = decay_rate

        # Color schemes (BGR)
        self.color_schemes = {
            "red_blue": {
                1: (255, 0, 0),   # ON: Blue (BGR)
                -1: (0, 0, 255)   # OFF: Red
            },
            "green_red": {
                1: (0, 255, 0),   # ON: Green
                -1: (0, 0, 255)   # OFF: Red
            },
            "white": {
                1: (255, 255, 255),  # ON: White
                -1: (255, 255, 255)  # OFF: White
            }
        }

    def create_video(self,
                     events: List[Dict],
                     original_frames: np.ndarray,
                     fps: float,
                     output_path: str,
                     show_progress: bool = True) -> None:
        """
        Create an event visualization video

        Args:
            events: List of events
            original_frames: Original video frames (T, H, W)
            fps: Output video frame rate
            output_path: Output video path
            show_progress: Whether to show a progress bar
        """
        print(f"\n{'=' * 70}")
        print(f"Creating event visualization video")
        print(f"{'=' * 70}")
        print(f"Output path: {output_path}")
        print(f"Resolution: {self.width}x{self.height}")
        print(f"FPS: {fps}")
        print(f"Accumulation window: {self.accumulation_time * 1000:.1f} ms")
        print(f"Style: {self.style}")
        print(f"Total events: {len(events)}")
        print(f"{'=' * 70}\n")

        # Video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps,
                              (self.width, self.height), True)

        num_frames = len(original_frames)
        frame_duration = 1.0 / fps

        # Event index (for faster search)
        event_idx = 0

        # Accumulation layer (for decay effect)
        event_layer = np.zeros((self.height, self.width, 3), dtype=np.float32)

        # Progress bar
        pbar = tqdm(total=num_frames, desc="Rendering frames") if show_progress else None

        for frame_idx in range(num_frames):
            # Time window for the current frame
            t_start = frame_idx * frame_duration
            t_end = t_start + self.accumulation_time

            # Original frame
            frame = original_frames[frame_idx]

            # Convert to color if grayscale
            if len(frame.shape) == 2:
                frame_color = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            else:
                frame_color = frame.copy()

            # Decay previous events
            if self.decay:
                event_layer *= self.decay_rate
            else:
                event_layer.fill(0)

            # Collect events in the current window
            window_events = []
            temp_idx = event_idx

            while temp_idx < len(events) and events[temp_idx]['t'] < t_end:
                if events[temp_idx]['t'] >= t_start:
                    window_events.append(events[temp_idx])
                temp_idx += 1

            # Advance the main index to the start of the window
            while event_idx < len(events) and events[event_idx]['t'] < t_start:
                event_idx += 1

            # Render events
            if self.style == "heatmap":
                event_layer = self._render_heatmap(window_events, event_layer)
            else:
                event_layer = self._render_overlay(window_events, event_layer)

            # Compose final frame
            vis_frame = self._blend_frames(frame_color, event_layer)

            # Add info text
            vis_frame = self._add_info_text(
                vis_frame, frame_idx, num_frames,
                len(window_events), t_start
            )

            # Write to video
            out.write(vis_frame)

            if pbar:
                pbar.update(1)

        if pbar:
            pbar.close()

        out.release()

        print(f"\n✅ Video created: {output_path}")
        print(f"{'=' * 70}\n")

    def _render_overlay(self,
                        events: List[Dict],
                        event_layer: np.ndarray) -> np.ndarray:
        """Render events in overlay mode"""
        colors = self.color_schemes[self.style]

        for event in events:
            x, y = event['x'], event['y']
            polarity = event['polarity']

            # Bounds check
            if 0 <= x < self.width and 0 <= y < self.height:
                color = colors[polarity]
                # Accumulate color (instead of overwrite)
                event_layer[y, x] = np.minimum(
                    event_layer[y, x] + np.array(color, dtype=np.float32),
                    255.0
                )

        return event_layer

    def _render_heatmap(self,
                        events: List[Dict],
                        event_layer: np.ndarray) -> np.ndarray:
        """Render heatmap mode"""
        # Build event density map
        density = np.zeros((self.height, self.width), dtype=np.float32)

        for event in events:
            x, y = event['x'], event['y']
            if 0 <= x < self.width and 0 <= y < self.height:
                density[y, x] += 1.0

        # Add to accumulation layer (use the blue channel as scratch)
        event_layer[:, :, 0] += density

        # Normalize and apply colormap
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
        Blend the original frame with the event layer

        Args:
            original: Original frame (BGR)
            event_layer: Event layer (float32, BGR)
            alpha: Event layer transparency
        """
        # Slightly darken the original frame so events pop out
        darkened = (original * 0.7).astype(np.uint8)

        # (Optional) mask if needed in future
        # mask = (event_layer.sum(axis=2) > 0).astype(np.float32)
        # mask = np.stack([mask] * 3, axis=2)

        # Blend
        event_uint8 = event_layer.astype(np.uint8)
        blended = cv2.addWeighted(darkened, 1.0, event_uint8, alpha, 0)

        return blended

    def _add_info_text(self,
                       frame: np.ndarray,
                       frame_idx: int,
                       total_frames: int,
                       num_events: int,
                       timestamp: float) -> np.ndarray:
        """Add information text onto the frame"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        color = (255, 255, 255)
        bg_color = (0, 0, 0)

        # Prepare text lines
        texts = [
            f"Frame: {frame_idx + 1}/{total_frames}",
            f"Time: {timestamp:.3f}s",
            f"Events: {num_events}"
        ]

        y_offset = 20
        for text in texts:
            # Measure text
            (text_width, text_height), _ = cv2.getTextSize(
                text, font, font_scale, thickness
            )

            # Draw background rectangle
            cv2.rectangle(frame,
                          (5, y_offset - text_height - 2),
                          (15 + text_width, y_offset + 2),
                          bg_color, -1)

            # Put text
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
        Create a side-by-side comparison video (Original | Events | Overlay)

        Args:
            events: List of events
            original_frames: Original frames
            fps: Frames per second
            output_path: Output path
        """
        print(f"\nCreating three-column comparison video...")

        # Three visualizers
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

        # Output width = 3 * original width
        out_width = self.width * 3
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps,
                              (out_width, self.height), True)

        num_frames = len(original_frames)
        frame_duration = 1.0 / fps
        event_idx = 0

        for frame_idx in tqdm(range(num_frames), desc="Rendering comparison"):
            t_start = frame_idx * frame_duration
            t_end = t_start + self.accumulation_time

            # Original frame
            frame = original_frames[frame_idx]
            if len(frame.shape) == 2:
                frame_color = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            else:
                frame_color = frame.copy()

            # Collect events in window
            window_events = []
            temp_idx = event_idx
            while temp_idx < len(events) and events[temp_idx]['t'] < t_end:
                if events[temp_idx]['t'] >= t_start:
                    window_events.append(events[temp_idx])
                temp_idx += 1

            while event_idx < len(events) and events[event_idx]['t'] < t_start:
                event_idx += 1

            # Events only (black background)
            event_only = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            event_layer = np.zeros((self.height, self.width, 3), dtype=np.float32)
            event_layer = viz_events_only._render_overlay(window_events, event_layer)
            event_only = event_layer.astype(np.uint8)

            # Overlay visualization
            event_layer2 = np.zeros((self.height, self.width, 3), dtype=np.float32)
            event_layer2 = viz_overlay._render_overlay(window_events, event_layer2)
            overlay = viz_overlay._blend_frames(frame_color, event_layer2)

            # Labels
            cv2.putText(frame_color, "Original", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(event_only, "Events Only", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(overlay, "Overlay", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Horizontal concat
            combined = np.hstack([frame_color, event_only, overlay])

            out.write(combined)

        out.release()
        print(f"✅ Comparison video created: {output_path}\n")


def create_event_gif(events: List[Dict],
                     width: int,
                     height: int,
                     duration: float,
                     output_path: str,
                     num_frames: int = 30):
    """
    Create an animated preview (GIF-like) of events

    Args:
        events: List of events
        width, height: Size
        duration: Total duration (seconds)
        output_path: Output path for GIF/video
        num_frames: Number of frames in the preview
    """
    print(f"\nCreating event preview animation...")

    frames = []
    time_step = duration / num_frames

    for i in range(num_frames):
        t_start = i * time_step
        t_end = t_start + time_step * 2  # Accumulate two time steps

        # Black background
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # Collect events in the window
        window_events = [e for e in events
                         if t_start <= e['t'] < t_end]

        # Draw events
        for event in window_events:
            x, y = event['x'], event['y']
            if 0 <= x < width and 0 <= y < height:
                color = (0, 255, 0) if event['polarity'] == 1 else (0, 0, 255)
                cv2.circle(frame, (x, y), 1, color, -1)

        frames.append(frame)

    # Using OpenCV here; for a true GIF, consider using PIL/imageio
    print(f"Preview frames: {len(frames)}")
    print(f"(Note: Additional libraries are required to write a real GIF; generating a video instead is recommended.)")


if __name__ == "__main__":
    print("Visualization module loaded")
    print("Usage example:")
    print("  from visualization import EventVisualizer")
    print("  viz = EventVisualizer(width, height)")
    print("  viz.create_video(events, frames, fps, 'output.mp4')")
