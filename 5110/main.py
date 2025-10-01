"""
main.py - Event Camera Simulator Main Program
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
    Load video file

    Args:
        video_path: Path to the input video file
        max_frames: Maximum number of frames to load (None = load all)

    Returns:
        (frames, fps, original_size)
    """
    print(f"\nLoading video: {video_path}")

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    # Retrieve video info
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("Video info:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Total frames: {total_frames}")
    print(f"  Duration: {total_frames / fps:.2f} seconds")

    # Read frames
    frames = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray)
        frame_count += 1

        # Stop if reaching max_frames
        if max_frames and frame_count >= max_frames:
            print(f"  (Limited to first {max_frames} frames)")
            break

    cap.release()

    frames = np.array(frames)
    print(f"Successfully loaded {len(frames)} frames")

    return frames, fps, (width, height)


def save_events(events, output_path: str):
    """
    Save event data

    Args:
        events: List of events
        output_path: Output file path (.txt or .npy)
    """
    print(f"\nSaving event data: {output_path}")

    ext = os.path.splitext(output_path)[1]

    if ext == '.txt':
        # Save as text: t x y polarity
        with open(output_path, 'w') as f:
            f.write("# t(s) x y polarity\n")
            for e in events:
                f.write(f"{e['t']:.6f} {e['x']} {e['y']} {e['polarity']}\n")

    elif ext == '.npy':
        # Save as NumPy binary (compact)
        event_array = np.array([
            (e['t'], e['x'], e['y'], e['polarity'])
            for e in events
        ], dtype=[('t', 'f8'), ('x', 'i4'), ('y', 'i4'), ('polarity', 'i4')])
        np.save(output_path, event_array)

    else:
        raise ValueError(f"Unsupported file format: {ext}, use .txt or .npy")

    file_size = os.path.getsize(output_path) / 1024 / 1024  # MB
    print(f"Saved successfully! File size: {file_size:.2f} MB")


def create_output_directory(base_dir: str = "output") -> str:
    """Create output directory"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, f"sim_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Event Camera Simulator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Input / Output
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Path to input video file')
    parser.add_argument('--output', '-o', type=str, default='output',
                        help='Output directory')
    parser.add_argument('--max-frames', type=int, default=None,
                        help='Maximum number of frames to process (for testing)')

    # Simulator parameters
    parser.add_argument('--threshold', '-C', type=float, default=0.3,
                        help='Contrast threshold')
    parser.add_argument('--time-resolution', type=float, default=0.0001,
                        help='Timestamp resolution (seconds)')
    parser.add_argument('--threshold-std', type=float, default=0.03,
                        help='Standard deviation of threshold mismatch noise')

    # Performance
    parser.add_argument('--multiprocessing', action='store_true',
                        help='Enable multiprocessing acceleration')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of worker processes (default = CPU cores)')

    # Output format
    parser.add_argument('--format', choices=['txt', 'npy'], default='txt',
                        help='Event data save format')

    # Visualization
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualization video')
    parser.add_argument('--viz-style', choices=['red_blue', 'green_red', 'white', 'heatmap'],
                        default='red_blue', help='Visualization style')
    parser.add_argument('--accumulation-time', type=float, default=0.033,
                        help='Event accumulation time (seconds), default 33ms')
    parser.add_argument('--comparison', action='store_true',
                        help='Generate 3-panel comparison video')
    parser.add_argument('--output-fps', type=float, default=None,
                        help='Output video FPS (default = input FPS)')

    args = parser.parse_args()

    # Print configuration
    print("\n" + "=" * 60)
    print("Event Camera Simulator")
    print("=" * 60)
    print("\nConfiguration:")
    print(f"  Input video: {args.input}")
    print(f"  Output directory: {args.output}")
    print(f"  Contrast threshold: {args.threshold}")
    print(f"  Time resolution: {args.time_resolution * 1000} ms")
    print(f"  Threshold noise: ±{args.threshold_std}")
    print(f"  Multiprocessing: {'Yes' if args.multiprocessing else 'No'}")
    if args.visualize:
        print(f"  Visualization: Yes ({args.viz_style})")
        print(f"  Accumulation time: {args.accumulation_time * 1000} ms")

    try:
        # Step 1: Load video
        frames, fps, size = load_video(args.input, args.max_frames)

        # Step 2: Create sensor
        sensor = EventCameraSensor(
            contrast_threshold=args.threshold,
            timestamp_resolution=args.time_resolution,
            threshold_std=args.threshold_std,
            use_multiprocessing=args.multiprocessing,
            num_workers=args.workers
        )

        # Step 3: Simulate events
        events = sensor.simulate(frames, fps, show_progress=True)

        # Step 4: Save results
        output_dir = create_output_directory(args.output)
        print(f"\nOutput directory: {output_dir}")

        event_file = os.path.join(output_dir, f"events.{args.format}")
        save_events(events, event_file)

        # Save config info
        config_file = os.path.join(output_dir, "config.txt")
        with open(config_file, 'w') as f:
            f.write("Event Camera Simulator Configuration\n")
            f.write("=" * 40 + "\n")
            f.write(f"Input video: {args.input}\n")
            f.write(f"Resolution: {size[0]}x{size[1]}\n")
            f.write(f"FPS: {fps}\n")
            f.write(f"Frames: {len(frames)}\n")
            f.write(f"Contrast threshold: {args.threshold}\n")
            f.write(f"Time resolution: {args.time_resolution} s\n")
            f.write(f"Threshold noise std: {args.threshold_std}\n")
            f.write(f"\nTotal generated events: {len(events)}\n")

        print(f"Configuration saved: {config_file}")

        # Step 5: Visualization
        if args.visualize and len(events) > 0:
            print(f"\n{'=' * 60}")
            print("Generating visualization video")
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
                viz_file = os.path.join(output_dir, "comparison.mp4")
                visualizer.create_comparison_video(events, frames, output_fps, viz_file)
            else:
                viz_file = os.path.join(output_dir, "visualization.mp4")
                visualizer.create_video(events, frames, output_fps, viz_file, show_progress=True)

            print(f"Visualization video saved: {viz_file}")
        elif args.visualize and len(events) == 0:
            print("\nWarning: Cannot generate visualization because no events were generated")

        print("\n" + "=" * 60)
        print("Simulation complete!")
        print("=" * 60)

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
