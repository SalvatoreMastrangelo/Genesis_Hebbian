"""Allow ``python -m WP2.multi_urdf_utils`` to launch the benchmark or render videos."""
import sys
import argparse

def main():
    """CLI dispatcher for multi_urdf_utils commands."""
    if len(sys.argv) > 1 and sys.argv[1] == "render-videos":
        from WP2.multi_urdf_utils.render_videos import main as render_main
        render_main()
    else:
        from WP2.multi_urdf_utils.run import main as benchmark_main
        benchmark_main()

if __name__ == "__main__":
    main()
