"""
pixel.py - Pixel-level event generation module
Implements the event detection logic for a single pixel
"""

import numpy as np
from scipy.interpolate import interp1d
from typing import List, Dict, Tuple


class PixelEventGenerator:
    """Single-pixel event generator"""

    def __init__(self,
                 contrast_threshold: float = 0.3,
                 timestamp_resolution: float = 0.0001,
                 threshold_std: float = 0.03):
        """
        Initialize the pixel event generator

        Args:
            contrast_threshold: Contrast threshold (C value)
            timestamp_resolution: Timestamp resolution (seconds)
            threshold_std: Standard deviation of threshold mismatch noise
        """
        self.base_threshold = contrast_threshold
        self.timestamp_resolution = timestamp_resolution
        self.threshold_std = threshold_std

        # Generate unique thresholds for this pixel (simulate mismatch noise)
        self.pos_threshold = self.base_threshold + np.random.normal(0, threshold_std)
        self.neg_threshold = -self.base_threshold + np.random.normal(0, threshold_std)

    def process(self,
                pixel_values: np.ndarray,
                frame_timestamps: np.ndarray,
                x: int,
                y: int) -> List[Dict]:
        """
        Process the time series of a single pixel and generate events

        Args:
            pixel_values: DN values of the pixel across frames (T,)
            frame_timestamps: Corresponding timestamps (T,)
            x, y: Pixel coordinates

        Returns:
            A list of events, each event is a dict with {x, y, t, polarity}
        """
        # Step 1: Convert DN values to log intensity
        log_intensity = self._to_log_intensity(pixel_values)

        # Step 2: Time interpolation to generate high-resolution time series
        interp_timestamps, interp_intensity = self._interpolate(
            frame_timestamps, log_intensity
        )

        # Step 3: Event detection
        events = self._detect_events(
            interp_intensity,
            interp_timestamps,
            x, y
        )

        return events

    def _to_log_intensity(self, pixel_values: np.ndarray) -> np.ndarray:
        """
        Convert DN values to log intensity

        Args:
            pixel_values: DN value array

        Returns:
            Log intensity array
        """
        # Add a small constant to avoid log(0)
        epsilon = 1e-3
        safe_values = np.maximum(pixel_values, epsilon)

        # Normalize to [0, 1] (assuming 8-bit image)
        normalized = safe_values / 255.0

        # Compute logarithm
        log_intensity = np.log(normalized + epsilon)

        return log_intensity

    def _interpolate(self,
                     timestamps: np.ndarray,
                     values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Interpolate the time series to produce high-resolution data

        Args:
            timestamps: Original timestamps
            values: Original values

        Returns:
            (Interpolated timestamps, Interpolated values)
        """
        # Create a linear interpolation function
        f = interp1d(timestamps, values, kind='linear',
                     fill_value='extrapolate', assume_sorted=True)

        # Generate high-resolution timestamps
        t_start = timestamps[0]
        t_end = timestamps[-1]
        high_res_timestamps = np.arange(t_start, t_end, self.timestamp_resolution)

        # Ensure the last timestamp is included
        if high_res_timestamps[-1] < t_end:
            high_res_timestamps = np.append(high_res_timestamps, t_end)

        # Perform interpolation
        interpolated_values = f(high_res_timestamps)

        return high_res_timestamps, interpolated_values

    def _detect_events(self,
                       log_intensity: np.ndarray,
                       timestamps: np.ndarray,
                       x: int,
                       y: int) -> List[Dict]:
        """
        Detect events

        Args:
            log_intensity: Log intensity series
            timestamps: Timestamp series
            x, y: Pixel coordinates

        Returns:
            A list of detected events
        """
        events = []

        # Initialize reference intensity
        reference_intensity = log_intensity[0]

        for i in range(1, len(log_intensity)):
            current_intensity = log_intensity[i]
            delta_log_I = current_intensity - reference_intensity

            # Detect positive events (ON / brightness increase)
            if delta_log_I >= self.pos_threshold:
                events.append({
                    'x': x,
                    'y': y,
                    't': timestamps[i],
                    'polarity': 1  # ON event
                })
                # Update reference intensity
                reference_intensity = current_intensity

            # Detect negative events (OFF / brightness decrease)
            elif delta_log_I <= self.neg_threshold:
                events.append({
                    'x': x,
                    'y': y,
                    't': timestamps[i],
                    'polarity': -1  # OFF event
                })
                # Update reference intensity
                reference_intensity = current_intensity

        return events


def test_pixel_generator():
    """Test function: simulate brightness changes for a single pixel"""
    print("=== Test: Single Pixel Event Generation ===\n")

    # Simulated data: 30 frames, brightness increases then decreases
    num_frames = 30
    fps = 30
    timestamps = np.arange(num_frames) / fps  # 0 to 1 second

    # Simulate brightness change: sinusoidal wave
    pixel_values = 128 + 100 * np.sin(2 * np.pi * timestamps)
    pixel_values = pixel_values.astype(np.uint8)

    print(f"Input: {num_frames} frames, FPS={fps}")
    print(f"Time range: {timestamps[0]:.3f}s - {timestamps[-1]:.3f}s")
    print(f"Brightness range: {pixel_values.min()} - {pixel_values.max()}\n")

    # Create generator
    generator = PixelEventGenerator(
        contrast_threshold=0.2,
        timestamp_resolution=0.001  # 1ms
    )

    # Generate events
    events = generator.process(pixel_values, timestamps, x=10, y=20)

    print(f"Generated events: {len(events)}")
    print(f"\nFirst 5 events:")
    for i, event in enumerate(events[:5]):
        polarity_str = "ON " if event['polarity'] == 1 else "OFF"
        print(f"  {i + 1}. t={event['t']:.4f}s, ({event['x']},{event['y']}), {polarity_str}")

    # Statistics
    on_events = sum(1 for e in events if e['polarity'] == 1)
    off_events = sum(1 for e in events if e['polarity'] == -1)
    print(f"\nEvent statistics:")
    print(f"  ON events:  {on_events}")
    print(f"  OFF events: {off_events}")
    print(f"  Total:      {len(events)}")


if __name__ == "__main__":
    test_pixel_generator()
