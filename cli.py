#!/usr/bin/env python3
"""
🎛️ YT STUDIO MASTER - Command Line Interface

Simplified CLI wrapper for the consolidated yt_master_core.py system.

Usage:
    python cli.py "https://youtube.com/watch?v=..." -p club
    python cli.py "audio_file.wav" -p festival -v
    python cli.py url1 url2 url3 --batch
"""

import argparse
import logging
import sys

from yt_master_core import (
    VSTPluginManager,
    logger,
    process_batch,
    process_single,
)


def main():
    parser = argparse.ArgumentParser(
        description="🎛️ YT Master - Professional Audio Processing Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "https://youtube.com/watch?v=dQw4w9WgXcQ" -p club
  %(prog)s "audio.wav" -p festival -v
  %(prog)s url1.txt url2.txt --batch -p streaming
  %(prog)s "poor_quality.mp3" --force-quality -p club
        """
    )

    # Input options
    parser.add_argument("input", nargs="+", help="YouTube URL(s) or audio file(s)")

    # Output options
    parser.add_argument("-o", "--output", default="output", help="Output directory (default: output)")

    # Processing options
    parser.add_argument(
        "-p",
        "--preset",
        choices=["club", "festival", "radio", "streaming", "dynamic", "reference"],
        default="club",
        help="Mastering preset (default: club). Each preset optimized for specific use cases."
    )

    parser.add_argument(
        "--force-quality", 
        action="store_true", 
        help="Enable maximum processing for challenging sources (slower but better quality)"
    )

    parser.add_argument(
        "--batch", 
        action="store_true", 
        help="Process multiple files in parallel for efficiency"
    )

    # Logging options
    parser.add_argument("-v", "--verbose", action="store_true", help="Detailed processing information")
    parser.add_argument("-q", "--quiet", action="store_true", help="Minimal output (errors only)")

    args = parser.parse_args()

    # Configure logging based on verbosity
    if args.quiet:
        logging.getLogger().setLevel(logging.ERROR)
    elif args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)

    # Print startup banner
    if not args.quiet:
        print("🎛️ YT STUDIO MASTER - PROFESSIONAL EDITION")
        print("=" * 50)
        print(f"📁 Output: {args.output}")
        print(f"🎯 Preset: {args.preset.upper()}")
        if args.force_quality:
            print("🔥 Force Quality Mode: ENABLED")
        print("=" * 50)

    try:
        # Initialize VST Manager
        vst_manager = VSTPluginManager()
        
        if args.batch or len(args.input) > 1:
            # Batch processing
            logger.info(f"🚀 Starting batch processing of {len(args.input)} items...")
            results = process_batch(
                args.input,
                output_dir=args.output,
                preset=args.preset,
                force_quality=args.force_quality,
                vst_manager=vst_manager
            )
            
            # Summary
            if isinstance(results, list):
                successful = sum(1 for success, _ in results if success)
                total = len(results)
                logger.info(f"✅ Batch processing complete: {successful}/{total} successful")
            else:
                logger.info("✅ Batch processing complete")
            
        else:
            # Single file processing
            success, metadata = process_single(
                args.input[0],
                output_dir=args.output,
                preset=args.preset,
                force_quality=args.force_quality,
                vst_manager=vst_manager
            )
            
            if success:
                logger.info("✅ Processing completed successfully!")
                if not args.quiet and metadata and isinstance(metadata, dict):
                    print(f"📈 Final Quality: {metadata.get('quality_score', 'N/A')}/100")
                    print(f"🎚️ LUFS: {metadata.get('lufs', 'N/A')}")
            else:
                logger.error("❌ Processing failed!")
                sys.exit(1)

    except KeyboardInterrupt:
        logger.info("\n⏹️ Processing interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
