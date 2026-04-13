"""
Object Tracker Demo

Demonstrates the ObjectTracker class for detecting objects with affordances
and interaction points.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
import numpy as np
import cv2
from dotenv import load_dotenv

# Load environment
load_dotenv()


def get_test_image():
    """Create or load test image."""
    # Try to get RealSense camera using reusable camera class
    try:
        from src.camera import RealSenseCamera

        print("Starting RealSense camera...")
        camera = RealSenseCamera(
            width=640,
            height=480,
            fps=30,
            enable_depth=True,
            auto_start=True
        )

        # Wait for stable frames
        print("  Warming up camera (30 frames)...")
        for _ in range(30):
            camera.capture_frame()

        # Get aligned color and depth frames
        color_image, depth_image = camera.get_aligned_frames()

        # Get intrinsics
        intrinsics = camera.get_camera_intrinsics()

        # Stop camera
        camera.stop()

        print(f"✓ Captured frame from RealSense: {color_image.shape}")
        return color_image, depth_image, intrinsics

    except ImportError as e:
        print(f"⚠ RealSense module not available: {e}")
        print("   Install pyrealsense2 to use live camera")
        return None, None, None
    except RuntimeError as e:
        print(f"⚠ RealSense camera error: {e}")
        print("   Check that camera is connected and not in use by another process")
        return None, None, None
    except Exception as e:
        print(f"⚠ Unexpected error accessing RealSense: {type(e).__name__}")
        print(f"   Error details: {e}")
        return None, None, None


def visualize_detections(
    image: np.ndarray,
    objects: list,
    show_interaction_points: bool = True
):
    """
    Visualize detected objects with affordances and interaction points.

    Args:
        image: RGB image
        objects: List of DetectedObject instances
        show_interaction_points: Whether to show interaction points
    """
    vis_image = image.copy()
    height, width = vis_image.shape[:2]

    # Define colors for different affordances
    affordance_colors = {
        "graspable": (0, 255, 0),      # Green
        "pourable": (255, 0, 0),       # Blue
        "containable": (0, 0, 255),    # Red
        "pushable": (255, 255, 0),     # Cyan
        "pullable": (255, 0, 255),     # Magenta
        "openable": (0, 255, 255),     # Yellow
        "supportable": (128, 128, 128) # Gray
    }

    for obj in objects:
        # Draw object center
        if obj.position_2d:
            y_norm, x_norm = obj.position_2d
            center_y = int((y_norm / 1000.0) * height)
            center_x = int((x_norm / 1000.0) * width)

            # Draw center point
            cv2.circle(vis_image, (center_x, center_y), 8, (0, 0, 255), -1)

            # Draw label
            label = f"{obj.object_id}"
            if obj.position_3d is not None:
                dist = obj.position_3d[2]  # z distance
                label += f" ({dist:.2f}m)"

            cv2.putText(
                vis_image,
                label,
                (center_x + 10, center_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2
            )

        # Draw bounding box
        if obj.bounding_box_2d:
            y1, x1, y2, x2 = obj.bounding_box_2d
            pt1 = (int((x1 / 1000.0) * width), int((y1 / 1000.0) * height))
            pt2 = (int((x2 / 1000.0) * width), int((y2 / 1000.0) * height))
            cv2.rectangle(vis_image, pt1, pt2, (0, 255, 0), 2)

        # Draw interaction points
        if show_interaction_points and obj.interaction_points:
            for affordance, point in obj.interaction_points.items():
                y_norm, x_norm = point.position_2d
                point_y = int((y_norm / 1000.0) * height)
                point_x = int((x_norm / 1000.0) * width)

                # Get color for this affordance
                color = affordance_colors.get(affordance, (255, 255, 255))

                # Draw interaction point
                cv2.drawMarker(
                    vis_image,
                    (point_x, point_y),
                    color,
                    markerType=cv2.MARKER_CROSS,
                    markerSize=20,
                    thickness=2
                )

                # Draw label
                label = affordance[:4]  # Abbreviate
                cv2.putText(
                    vis_image,
                    label,
                    (point_x + 12, point_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    color,
                    1
                )

    return vis_image


def print_detection_results(objects: list):
    """Print detailed detection results."""
    print("\n" + "=" * 60)
    print("DETECTION RESULTS")
    print("=" * 60)

    if not objects:
        print("No objects detected")
        return

    for i, obj in enumerate(objects, 1):
        print(f"\n{i}. {obj.object_id} ({obj.object_type})")

        # Position
        if obj.position_2d:
            print(f"   Position 2D: {obj.position_2d}")
        if obj.position_3d is not None:
            x, y, z = obj.position_3d
            print(f"   Position 3D: [{x:.3f}, {y:.3f}, {z:.3f}] m")

        # Affordances
        print(f"   Affordances ({len(obj.affordances)}):")
        for affordance in sorted(obj.affordances):
            print(f"      • {affordance}", end="")

            # Show interaction point if available
            if affordance in obj.interaction_points:
                point = obj.interaction_points[affordance]
                print(f" → {point.position_2d}")
            else:
                print()


async def main():
    """Run object tracker demo."""
    print("=" * 60)
    print("OBJECT TRACKER DEMO")
    print("=" * 60)
    print()

    # Check API key
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("⚠ GEMINI_API_KEY or GOOGLE_API_KEY not set")
        print("  Please set it in .env file or environment")
        return

    # Initialize tracker
    print("Initializing ObjectTracker...")
    from src.perception import ObjectTracker

    try:
        tracker = ObjectTracker(
            api_key=api_key,
            model_name="auto",
            thinking_budget=0,
            max_parallel_requests=5
        )
        print("✓ ObjectTracker initialized successfully")
        print(f"  Model: {tracker.model_name}")
        print(f"  Max parallel requests: {tracker.max_parallel_requests}")
    except Exception as e:
        print(f"\n✗ ERROR: Failed to initialize ObjectTracker!")
        print(f"   Error type: {type(e).__name__}")
        print(f"   Error message: {str(e)}")
        print(f"\n   This could be due to:")
        print(f"   - Invalid API key format")
        print(f"   - Missing configuration files")
        print(f"   - Import errors")
        return
    print()

    # Get image
    print("Getting test image...")
    color_image, depth_image, intrinsics = get_test_image()

    if color_image is None:
        print("⚠ No camera available, using synthetic image")
        # Create simple test image
        color_image = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(color_image, (200, 150), (300, 350), (0, 0, 255), -1)  # Red rectangle
        cv2.circle(color_image, (450, 240), 50, (0, 255, 0), -1)  # Green circle
        cv2.putText(
            color_image,
            "TEST SCENE",
            (220, 400),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )

    print(f"Image size: {color_image.shape}")
    print()

    # Show input image
    cv2.imshow("Input Image", cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR))
    print("Press any key to start detection...")
    cv2.waitKey(0)

    # Detect objects
    print("\n" + "=" * 60)
    print("DETECTING OBJECTS")
    print("=" * 60)

    try:
        objects = await tracker.detect_objects(
            color_image,
            depth_image,
            intrinsics
        )
    except Exception as e:
        print(f"\n✗ ERROR: Object detection failed!")
        print(f"   Error type: {type(e).__name__}")
        print(f"   Error message: {str(e)}")
        print(f"\n   This could be due to:")
        print(f"   - Invalid API key (check GEMINI_API_KEY)")
        print(f"   - Network connectivity issues")
        print(f"   - Invalid image format")
        print(f"   - API rate limiting")
        print(f"\n   Check the logs above for more details.")

        # Clean up and exit
        await tracker.aclose()
        cv2.destroyAllWindows()
        return

    # Print results
    print_detection_results(objects)

    # Log detection summary
    if objects:
        print(f"\n✓ Successfully detected {len(objects)} object(s)")
        total_affordances = sum(len(obj.affordances) for obj in objects)
        total_interaction_points = sum(len(obj.interaction_points) for obj in objects)
        print(f"  Total affordances: {total_affordances}")
        print(f"  Total interaction points: {total_interaction_points}")
    else:
        print("\n⚠ WARNING: No objects detected!")
        print("   Possible reasons:")
        print("   - Scene is empty or too dark")
        print("   - Objects are too far from camera")
        print("   - VLM failed to recognize objects")
        print("   - Try adjusting lighting or camera position")

    # Save detections prompt
    if objects:
        print("\n" + "=" * 60)
        save = input("Save detections to JSON? (y/n): ").strip().lower()
        if save == 'y':
            output_path = tracker.save_detections("detections.json", include_timestamp=True)

    # Visualize
    if objects:
        print("\n" + "=" * 60)
        print("VISUALIZATION")
        print("=" * 60)
        print("Generating visualization...")

        vis_image = visualize_detections(color_image, objects)

        cv2.imshow("Detection Results", cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
        print("✓ Showing visualization (press any key to continue)")
        cv2.waitKey(0)

        # Optionally save
        save = input("\nSave visualization? (y/n): ").strip().lower()
        if save == 'y':
            output_path = "output_object_tracking.png"
            cv2.imwrite(
                output_path,
                cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
            )
            print(f"✓ Saved to {output_path}")

    # Interactive query
    print("\n" + "=" * 60)
    print("INTERACTIVE QUERIES")
    print("=" * 60)

    if objects:
        print("\nAvailable queries:")
        print("  1. Get objects with specific affordance")
        print("  2. Update interaction point with task context")
        print("  3. Show all objects by type")
        print("  4. Load detections from JSON")
        print("  0. Exit")

        while True:
            choice = input("\nSelect query (0-4): ").strip()

            if choice == "0":
                break

            elif choice == "1":
                affordance = input("Enter affordance (e.g., graspable, pourable): ").strip()
                matching = tracker.get_objects_with_affordance(affordance)
                print(f"\nObjects with '{affordance}' affordance:")
                for obj in matching:
                    print(f"  • {obj.object_id}")

            elif choice == "2":
                print("\nAvailable objects:")
                for i, obj in enumerate(objects, 1):
                    print(f"  {i}. {obj.object_id}")

                obj_idx = int(input("Select object (number): ").strip()) - 1
                if 0 <= obj_idx < len(objects):
                    obj = objects[obj_idx]

                    print(f"\nAffordances for {obj.object_id}:")
                    affordances_list = sorted(obj.affordances)
                    for i, aff in enumerate(affordances_list, 1):
                        print(f"  {i}. {aff}")

                    aff_idx = int(input("Select affordance (number): ").strip()) - 1
                    if 0 <= aff_idx < len(affordances_list):
                        affordance = affordances_list[aff_idx]

                        task_context = input("Enter task context (or Enter to skip): ").strip()
                        if not task_context:
                            task_context = None

                        print(f"\nUpdating {affordance} point for {obj.object_id}...")
                        point = await tracker.update_interaction_point(
                            obj.object_id,
                            affordance,
                            task_context
                        )

                        if point:
                            print(f"✓ Updated interaction point:")
                            print(f"   Position: {point.position_2d}")
                            print(f"   Confidence: {point.confidence:.2f}")
                            print(f"   Reasoning: {point.reasoning}")

            elif choice == "3":
                object_types = set(obj.object_type for obj in objects)
                print("\nObjects by type:")
                for obj_type in sorted(object_types):
                    matching = tracker.get_objects_by_type(obj_type)
                    print(f"\n  {obj_type} ({len(matching)}):")
                    for obj in matching:
                        print(f"    • {obj.object_id}")

            elif choice == "4":
                file_path = input("Enter JSON file path: ").strip()
                try:
                    loaded_objects = tracker.load_detections(file_path)
                    print(f"\n✓ Loaded {len(loaded_objects)} objects")
                    print_detection_results(loaded_objects)

                    # Visualize loaded objects
                    visualize = input("\nVisualize loaded objects? (y/n): ").strip().lower()
                    if visualize == 'y' and color_image is not None:
                        vis_image = visualize_detections(color_image, loaded_objects)
                        cv2.imshow("Loaded Objects", cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
                        print("Press any key to close...")
                        cv2.waitKey(0)
                except Exception as e:
                    print(f"⚠ Failed to load detections: {e}")

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)

    # Clean up async resources
    await tracker.aclose()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    asyncio.run(main())
