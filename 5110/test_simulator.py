"""
test_simulator.py - Comprehensive simulator test script
Generates synthetic test videos and runs the simulator
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
    Create a synthetic test video

    Args:
        output_path: Output video path
        num_frames: Number of frames
        fps: Frame rate
        resolution: Resolution (height, width)
        pattern: Motion pattern ('moving_square', 'rotating', 'flashing')
    """
    height, width = resolution

    print(f"\nCreating synthetic test video: {pattern}")
    print(f"  Resolution: {width}x{height}")
    print(f"  Frames: {num_frames}")
    print(f"  Frame rate: {fps} FPS")

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), False)

    frames = []

    for i in range(num_frames):
        frame = np.ones((height, width), dtype=np.uint8) * 50  # Background

        if pattern == "moving_square":
            # Square moving left to right
            x_pos = int(20 + (width - 80) * i / num_frames)
            y_pos = height // 2 - 20
            frame[y_pos:y_pos + 40, x_pos:x_pos + 40] = 200

        elif pattern == "rotating":
            # Rotating line
            angle = 360 * i / num_frames
            center = (width // 2, height // 2)
            length = min(width, height) // 3
            end_x = int(center[0] + length * np.cos(np.radians(angle)))
            end_y = int(center[1] + length * np.sin(np.radians(angle)))
            cv2.line(frame, center, (end_x, end_y), 200, 3)

        elif pattern == "flashing":
            # Flashing circle
            if (i // 10) % 2 == 0:  # Toggle every 10 frames
                cv2.circle(frame, (width // 2, height // 2), 30, 200, -1)

        elif pattern == "expanding_circle":
            # Expanding circle
            radius = int(10 + 50 * (i % 30) / 30)
            cv2.circle(frame, (width // 2, height // 2), radius, 200, 2)

        frames.append(frame)
        out.write(frame)

    out.release()
    print(f"  Video saved: {output_path}")

    return np.array(frames), fps


def run_test(video_path: str,
             contrast_threshold: float = 0.2,
             use_multiprocessing: bool = False):
    """
    Run simulator test

    Args:
        video_path: Path to the video
        contrast_threshold: Contrast threshold
        use_multiprocessing: Whether to use multiprocessing
    """
    print("\n" + "=" * 70)
    print("Running Event Camera Simulator Test")
    print("=" * 70)

    # Load video
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

    print(f"\nVideo info:")
    print(f"  Frames: {len(frames)}")
    print(f"  Resolution: {frames.shape[2]}x{frames.shape[1]}")
    print(f"  Frame rate: {fps} FPS")

    # Create sensor
    sensor = EventCameraSensor(
        contrast_threshold=contrast_threshold,
        timestamp_resolution=0.001,  # 1 ms
        threshold_std=0.02,
        use_multiprocessing=use_multiprocessing
    )

    # Run simulation
    events = sensor.simulate(frames, fps, show_progress=True)

    # Analyze results
    if len(events) > 0:
        print("\n" + "=" * 70)
        print("Test Result Analysis")
        print("=" * 70)

        # Temporal distribution
        time_span = events[-1]['t'] - events[0]['t']
        print(f"\nTime statistics:")
        print(f"  Time span: {time_span:.3f} s")
        print(f"  Average event rate: {len(events) / time_span:.1f} events/sec")

        # Polarity distribution
        on_count = sum(1 for e in events if e['polarity'] == 1)
        off_count = len(events) - on_count
        print(f"\nPolarity distribution:")
        print(f"  ON events:  {on_count:6d} ({on_count / len(events) * 100:.1f}%)")
        print(f"  OFF events: {off_count:6d} ({off_count / len(events) * 100:.1f}%)")

        # Spatial distribution heatmap (stats only here)
        x_coords = np.array([e['x'] for e in events])
        y_coords = np.array([e['y'] for e in events])
        print(f"\nSpatial statistics:")
        print(f"  X range: [{x_coords.min()}, {x_coords.max()}]")
        print(f"  Y range: [{y_coords.min()}, {y_coords.max()}]")
        print(f"  Active pixels: {len(set(zip(x_coords, y_coords)))}")

        # Sample events
        print(f"\nEvent samples (first 10):")
        for i, e in enumerate(events[:10]):
            pol_str = "ON " if e['polarity'] == 1 else "OFF"
            print(f"  {i + 1:2d}. t={e['t']:.4f}s, ({e['x']:3d},{e['y']:3d}), {pol_str}")

        print("\n✅ Test passed!")
    else:
        print("\n❌ Warning: No events were generated!")
        print("   Suggestion: Lower the contrast threshold or check the input video")

    print("=" * 70 + "\n")

    return events


def test_all_patterns():
    """Test all motion patterns"""
    print("\n" + "=" * 70)
    print("Comprehensive Test: All Motion Patterns")
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
        print(f"Testing pattern: {pattern}")
        print(f"{'=' * 70}")

        # Create test video
        video_path = os.path.join(test_dir, f"{pattern}.mp4")
        frames, fps = create_synthetic_video(
            video_path,
            num_frames=60,
            fps=30.0,
            resolution=(180, 240),
            pattern=pattern
        )

        # Run test
        events = run_test(video_path, contrast_threshold=0.15)

        results[pattern] = {
            'num_events': len(events),
            'video_path': video_path
        }

    # Summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    for pattern, result in results.items():
        print(f"{pattern:20s}: {result['num_events']:6d} events")
    print("=" * 70 + "\n")


def quick_test():
    """Quick test: single simple scene"""
    print("\n" + "=" * 70)
    print("Quick Test")
    print("=" * 70)

    test_dir = "test_videos"
    os.makedirs(test_dir, exist_ok=True)

    video_path = os.path.join(test_dir, "quick_test.mp4")

    # Create a simple test video
    frames, fps = create_synthetic_video(
        video_path,
        num_frames=60,
        fps=30.0,
        resolution=(120, 160),
        pattern="moving_square"
    )

    # Run test
    events = run_test(video_path, contrast_threshold=0.2)

    return events


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "all":
        # Run tests for all patterns
        test_all_patterns()
    else:
        # Quick test
        quick_test()

        print("\nTip: Run 'python test_simulator.py all' to test all patterns")
