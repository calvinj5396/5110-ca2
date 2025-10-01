"""
demo.py - Full demonstration script
Showcases all functionalities of the event camera simulator
"""

import numpy as np
import cv2
import os
from sensor import EventCameraSensor
from visualization import EventVisualizer


def create_demo_video(output_dir="demo_output"):
    """Create a full demonstration"""

    print("\n" + "=" * 70)
    print("Event Camera Simulator - Full Demonstration")
    print("=" * 70)

    os.makedirs(output_dir, exist_ok=True)

    # ===== Scene 1: Moving Square =====
    print("\nScene 1: Moving Square")
    print("-" * 70)

    scene1_frames, fps = create_moving_square(
        num_frames=90,
        fps=30.0,
        resolution=(240, 320)
    )

    # Save input video
    scene1_input = os.path.join(output_dir, "scene1_input.mp4")
    save_video(scene1_frames, fps, scene1_input)

    # Simulate events
    sensor = EventCameraSensor(
        contrast_threshold=0.2,
        timestamp_resolution=0.001,
        threshold_std=0.02
    )
    events1 = sensor.simulate(scene1_frames, fps)

    # Visualization in multiple styles
    styles = ['red_blue', 'green_red', 'heatmap']

    for style in styles:
        print(f"\n  Generating {style} visualization...")
        viz = EventVisualizer(
            width=320,
            height=240,
            accumulation_time=0.033,
            style=style,
            decay=True
        )

        output_path = os.path.join(output_dir, f"scene1_{style}.mp4")
        viz.create_video(events1, scene1_frames, fps, output_path, show_progress=False)

    # Comparison video
    print("\n  Generating three-column comparison video...")
    viz_comp = EventVisualizer(320, 240, style='red_blue')
    comp_path = os.path.join(output_dir, "scene1_comparison.mp4")
    viz_comp.create_comparison_video(events1, scene1_frames, fps, comp_path)

    # ===== Scene 2: Rotating Line =====
    print("\nScene 2: Rotating Line")
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

    # ===== Scene 3: Multiple Moving Objects =====
    print("\nScene 3: Multiple Moving Objects")
    print("-" * 70)

    scene3_frames = create_multiple_objects(
        num_frames=120,
        resolution=(240, 320)
    )

    scene3_input = os.path.join(output_dir, "scene3_input.mp4")
    save_video(scene3_frames, fps, scene3_input)

    # Use lower threshold to capture more events
    sensor_sensitive = EventCameraSensor(
        contrast_threshold=0.15,
        timestamp_resolution=0.001,
        threshold_std=0.02
    )
    events3 = sensor_sensitive.simulate(scene3_frames, fps)

    viz3 = EventVisualizer(320, 240, style='red_blue', accumulation_time=0.05, decay=True)
    scene3_viz = os.path.join(output_dir, "scene3_visualization.mp4")
    viz3.create_video(events3, scene3_frames, fps, scene3_viz, show_progress=False)

    # ===== Generate Report =====
    print("\nGenerating report...")
    report_path = os.path.join(output_dir, "demo_report.txt")

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("Event Camera Simulator - Demonstration Report\n")
        f.write("=" * 70 + "\n\n")

        f.write("Scene 1: Moving Square\n")
        f.write(f"  Total events: {len(events1):,}\n")
        f.write(f"  ON events: {sum(1 for e in events1 if e['polarity'] == 1):,}\n")
        f.write(f"  OFF events: {sum(1 for e in events1 if e['polarity'] == -1):,}\n")
        f.write(f"  Output videos: scene1_*.mp4\n\n")

        f.write("Scene 2: Rotating Line\n")
        f.write(f"  Total events: {len(events2):,}\n")
        f.write(f"  ON events: {sum(1 for e in events2 if e['polarity'] == 1):,}\n")
        f.write(f"  OFF events: {sum(1 for e in events2 if e['polarity'] == -1):,}\n")
        f.write(f"  Output video: scene2_visualization.mp4\n\n")

        f.write("Scene 3: Multiple Moving Objects\n")
        f.write(f"  Total events: {len(events3):,}\n")
        f.write(f"  ON events: {sum(1 for e in events3 if e['polarity'] == 1):,}\n")
        f.write(f"  OFF events: {sum(1 for e in events3 if e['polarity'] == -1):,}\n")
        f.write(f"  Output video: scene3_visualization.mp4\n\n")

        f.write("Visualization styles:\n")
        f.write("  - red_blue: ON=Blue, OFF=Red\n")
        f.write("  - green_red: ON=Green, OFF=Red\n")
        f.write("  - heatmap: Event density heatmap\n\n")

        f.write("Comparison video:\n")
        f.write("  scene1_comparison.mp4 shows three-column comparison\n")

    print(f"Report saved: {report_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("Demonstration completed!")
    print("=" * 70)
    print(f"Output directory: {output_dir}/")
    print(f"\nGenerated files:")
    for filename in sorted(os.listdir(output_dir)):
        filepath = os.path.join(output_dir, filename)
        size_mb = os.path.getsize(filepath) / 1024 / 1024
        print(f"  {filename:40s} ({size_mb:6.2f} MB)")
    print("=" * 70 + "\n")


def create_moving_square(num_frames, fps, resolution):
    """Create a moving square scene"""
    height, width = resolution
    frames = []

    for i in range(num_frames):
        frame = np.ones((height, width), dtype=np.uint8) * 50

        # Square moves left to right
        x_pos = int(20 + (width - 80) * i / num_frames)
        y_pos = height // 2 - 20

        frame[y_pos:y_pos + 40, x_pos:x_pos + 40] = 200
        frames.append(frame)

    return np.array(frames), fps


def create_rotating_line(num_frames, resolution):
    """Create a rotating line scene"""
    height, width = resolution
    frames = []

    for i in range(num_frames):
        frame = np.ones((height, width), dtype=np.uint8) * 50

        angle = 360 * i / num_frames
        center = (width // 2, height // 2)
        length = min(width, height) // 3

        end_x = int(center[0] + length * np.cos(np.radians(angle)))
        end_y = int(center[1] + length * np.sin(np.radians(angle)))

        cv2.line(frame, center, (end_x, end_y), 220, 3)
        frames.append(frame)

    return np.array(frames)


def create_multiple_objects(num_frames, resolution):
    """Create a scene with multiple moving objects"""
    height, width = resolution
    frames = []

    for i in range(num_frames):
        frame = np.ones((height, width), dtype=np.uint8) * 50

        # Object 1: Horizontally moving circle
        x1 = int(30 + (width - 60) * i / num_frames)
        y1 = height // 3
        cv2.circle(frame, (x1, y1), 15, 200, -1)

        # Object 2: Vertically moving square
        x2 = width // 2
        y2 = int(30 + (height - 60) * i / num_frames)
        frame[y2 - 15:y2 + 15, x2 - 15:x2 + 15] = 180

        # Object 3: Diagonally moving triangle
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
    """Save frames as a video"""
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), False)

    for frame in frames:
        out.write(frame)

    out.release()


def create_presentation_slides():
    """Create presentation slide materials"""
    print("\nCreating presentation slide materials...")

    slides_dir = "demo_output/slides"
    os.makedirs(slides_dir, exist_ok=True)

    print("  Tip: Use the generated videos to create slides")
    print("  Suggested slide contents:")
    print("    1. Project Introduction - What is an event camera")
    print("    2. Algorithm Principle - Log intensity change detection")
    print("    3. System Architecture - Pixel → Array → Visualization")
    print("    4. Demo Scene 1 - Moving Square")
    print("    5. Demo Scene 2 - Rotating Line")
    print("    6. Demo Scene 3 - Multiple Objects")
    print("    7. Visualization Comparison - Different styles")
    print("    8. Performance Analysis - Event statistics")
    print("    9. Future Improvements - Noise models, etc.")
    print("   10. Q&A")


def quick_demo():
    """Quick demo (small size, fast generation)"""
    print("\nQuick Demo Mode (for testing)")
    print("=" * 70)

    output_dir = "quick_demo"
    os.makedirs(output_dir, exist_ok=True)

    # Lower resolution, fewer frames
    frames, fps = create_moving_square(
        num_frames=30,
        fps=15.0,
        resolution=(120, 160)
    )

    input_path = os.path.join(output_dir, "input.mp4")
    save_video(frames, fps, input_path)

    sensor = EventCameraSensor(
        contrast_threshold=0.2,
        timestamp_resolution=0.001
    )
    events = sensor.simulate(frames, fps)

    viz = EventVisualizer(160, 120, style='red_blue')
    output_path = os.path.join(output_dir, "output.mp4")
    viz.create_video(events, frames, fps, output_path)

    print(f"\n✅ Quick demo completed!")
    print(f"   Input: {input_path}")
    print(f"   Output: {output_path}")
    print(f"   Events: {len(events)}\n")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "quick":
            quick_demo()
        elif sys.argv[1] == "slides":
            create_presentation_slides()
        else:
            print("Usage:")
            print("  python demo.py        - Full demonstration")
            print("  python demo.py quick  - Quick demo")
            print("  python demo.py slides - Presentation suggestions")
    else:
        create_demo_video()
