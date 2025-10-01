"""
sensor.py - Sensor array simulation module
Extend single-pixel logic to the entire image sensor
"""

import numpy as np
from typing import List, Dict, Tuple
from tqdm import tqdm
import multiprocessing as mp
from functools import partial


class EventCameraSensor:
    """Event camera sensor array simulator"""

    def __init__(self,
                 contrast_threshold: float = 0.3,
                 timestamp_resolution: float = 0.0001,
                 threshold_std: float = 0.03,
                 use_multiprocessing: bool = False,
                 num_workers: int = None):
        """
        Initialize the sensor

        Args:
            contrast_threshold: Contrast threshold
            timestamp_resolution: Timestamp resolution (seconds)
            threshold_std: Std dev of threshold mismatch noise
            use_multiprocessing: Whether to accelerate with multiprocessing
            num_workers: Number of worker processes (None = auto)
        """
        self.contrast_threshold = contrast_threshold
        self.timestamp_resolution = timestamp_resolution
        self.threshold_std = threshold_std
        self.use_multiprocessing = use_multiprocessing
        self.num_workers = num_workers or mp.cpu_count()

        # Lazy import to avoid circular dependency
        from pixel import PixelEventGenerator
        self.PixelEventGenerator = PixelEventGenerator

    def simulate(self,
                 video_frames: np.ndarray,
                 fps: float,
                 show_progress: bool = True) -> List[Dict]:
        """
        Simulate event generation for the entire sensor

        Args:
            video_frames: Video frames array, shape=(T, H, W)
            fps: Frame rate
            show_progress: Whether to show a progress bar

        Returns:
            List of events sorted by timestamp
        """
        num_frames, height, width = video_frames.shape

        # Generate frame timestamps
        frame_timestamps = np.arange(num_frames) / fps

        print(f"\n{'=' * 60}")
        print(f"Event Camera Sensor Simulation")
        print(f"{'=' * 60}")
        print(f"Resolution: {width} x {height}")
        print(f"Frames: {num_frames}")
        print(f"Frame rate: {fps} FPS")
        print(f"Duration: {frame_timestamps[-1]:.3f} s")
        print(f"Contrast threshold: {self.contrast_threshold}")
        print(f"Time resolution: {self.timestamp_resolution * 1000:.3f} ms")
        print(f"{'=' * 60}\n")

        # Choose processing method
        if self.use_multiprocessing and height * width > 1000:
            print(f"Using multiprocessing (workers={self.num_workers})")
            all_events = self._simulate_multiprocess(
                video_frames, frame_timestamps, height, width, show_progress
            )
        else:
            all_events = self._simulate_sequential(
                video_frames, frame_timestamps, height, width, show_progress
            )

        # Sort by timestamp
        print("Sorting events...")
        all_events.sort(key=lambda e: e['t'])

        # Statistics
        self._print_statistics(all_events, frame_timestamps[-1])

        return all_events

    def _simulate_sequential(self,
                             video_frames: np.ndarray,
                             frame_timestamps: np.ndarray,
                             height: int,
                             width: int,
                             show_progress: bool) -> List[Dict]:
        """Sequential processing (single process)"""
        all_events = []
        total_pixels = height * width

        # Progress bar
        pbar = tqdm(total=total_pixels, desc="Processing pixels") if show_progress else None

        for y in range(height):
            for x in range(width):
                # Extract the time series for this pixel
                pixel_values = video_frames[:, y, x]

                # Create pixel generator
                generator = self.PixelEventGenerator(
                    contrast_threshold=self.contrast_threshold,
                    timestamp_resolution=self.timestamp_resolution,
                    threshold_std=self.threshold_std
                )

                # Generate events
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
        """Parallel processing with multiprocessing"""
        # Prepare per-row data
        row_data = [(y, video_frames[:, y, :], frame_timestamps)
                    for y in range(height)]

        # Worker function with fixed parameters
        process_func = partial(
            _process_row_worker,
            contrast_threshold=self.contrast_threshold,
            timestamp_resolution=self.timestamp_resolution,
            threshold_std=self.threshold_std
        )

        # Parallel execution
        with mp.Pool(processes=self.num_workers) as pool:
            if show_progress:
                results = list(tqdm(
                    pool.imap(process_func, row_data),
                    total=height,
                    desc="Processing rows"
                ))
            else:
                results = pool.map(process_func, row_data)

        # Merge results
        all_events = []
        for row_events in results:
            all_events.extend(row_events)

        return all_events

    def _print_statistics(self, events: List[Dict], duration: float):
        """Print summary statistics"""
        print(f"\n{'=' * 60}")
        print(f"Simulation Completed - Statistics")
        print(f"{'=' * 60}")
        print(f"Total events: {len(events):,}")

        if len(events) > 0:
            on_events = sum(1 for e in events if e['polarity'] == 1)
            off_events = sum(1 for e in events if e['polarity'] == -1)

            print(f"ON events:  {on_events:,} ({on_events / len(events) * 100:.1f}%)")
            print(f"OFF events: {off_events:,} ({off_events / len(events) * 100:.1f}%)")
            print(f"Event rate: {len(events) / duration:.1f} events/sec")

            # Time range
            print(f"\nTime span:")
            print(f"  First event: {events[0]['t']:.6f} s")
            print(f"  Last event:  {events[-1]['t']:.6f} s")

            # Spatial distribution (sample first 1000 events)
            x_coords = [e['x'] for e in events[:1000]]
            y_coords = [e['y'] for e in events[:1000]]
            print(f"\nSpatial distribution (first 1000 events):")
            print(f"  X range: {min(x_coords)} - {max(x_coords)}")
            print(f"  Y range: {min(y_coords)} - {max(y_coords)}")
        else:
            print("Warning: No events were generated!")
            print("Suggestion: Lower the contrast threshold or check the input video")

        print(f"{'=' * 60}\n")


def _process_row_worker(row_data: Tuple,
                        contrast_threshold: float,
                        timestamp_resolution: float,
                        threshold_std: float) -> List[Dict]:
    """
    Multiprocessing worker: process one row of pixels

    Args:
        row_data: (y, row_frames, timestamps)
        Other args are the same as the sensor configuration

    Returns:
        All events from this row
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
    """Test the sensor simulation"""
    print("\n" + "=" * 60)
    print("Testing sensor array simulation")
    print("=" * 60)

    # Create a test video: a moving square
    num_frames = 60
    height, width = 128, 128
    fps = 30.0

    print(f"\nCreating test video: {width}x{height}, {num_frames} frames, {fps} FPS")

    video_frames = np.zeros((num_frames, height, width), dtype=np.uint8)

    # Add a moving bright square
    for i in range(num_frames):
        # Square moves left to right
        x_pos = int(20 + (width - 60) * i / num_frames)
        video_frames[i, 40:80, x_pos:x_pos + 20] = 200

    # Add static background
    video_frames[:, :, :] += 50  # Background brightness

    print("Video features: Moving bright square + static background")

    # Create sensor
    sensor = EventCameraSensor(
        contrast_threshold=0.2,
        timestamp_resolution=0.001,  # 1 ms
        threshold_std=0.02,
        use_multiprocessing=False  # Single process for small images
    )

    # Simulate
    events = sensor.simulate(video_frames, fps, show_progress=True)

    # Optionally print some events
    if len(events) > 0:
        print("\nFirst 10 events:")
        for i, e in enumerate(events[:10]):
            pol_str = "ON " if e['polarity'] == 1 else "OFF"
            print(f"  {i + 1}. t={e['t']:.4f}s, ({e['x']:3d},{e['y']:3d}), {pol_str}")

    return events, video_frames


if __name__ == "__main__":
    events, frames = test_sensor()
