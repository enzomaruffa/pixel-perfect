# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python-based image processing framework called "pixel-perfect" that applies sophisticated pixel-level transformations through a composable pipeline architecture. The system is designed to treat images as RGBA data with strict parameter validation using Pydantic.

## Development Commands

### Environment Setup
- Python version: 3.10+ (as specified in `.python-version`)
- This project uses `uv` for dependency management

### Package Management
```bash
# Add a new dependency
uv add <package_name>

# Install all dependencies
uv sync

# Run the application
uv run main.py
```

### Running the Application
```bash
uv run main.py
```

## Architecture Overview

The project follows a pipeline-based architecture with these core components:

### Core Classes
- **ImageContext**: Carries accumulated state through the pipeline (width, height, channels, dtype, warnings, metadata)
- **BaseOperation**: Abstract base for all operations with validate(), get_cache_key(), estimate_memory(), and apply() methods
- **Pipeline**: Main orchestrator that loads images, runs validation, manages caching, and executes operations

### Operation Categories
1. **Pixel-Level Operations**: PixelFilter, PixelMath, PixelSort
2. **Row Operations**: RowShift, RowStretch, RowRemove, RowShuffle
3. **Column Operations**: ColumnShift, ColumnStretch, ColumnMirror, ColumnWeave
4. **Block Operations**: BlockFilter, BlockShift, BlockRotate, BlockScramble (virtual grid manipulation)
5. **Geometric Operations**: GridWarp, PerspectiveStretch, RadialStretch
6. **Aspect Ratio Operations**: AspectStretch, AspectCrop, AspectPad
7. **Channel Operations**: ChannelSwap, ChannelIsolate, AlphaGenerator
8. **Pattern Operations**: Mosaic, Dither

### Key Design Principles
- **Fail Fast, Fail Clear**: Complete pipeline validation after image load, before processing
- **Composable Operations**: Simple operations combine into complex effects
- **Channel Aware**: Full RGBA support with operations declaring channel requirements
- **Cache Smart**: Automatic per-operation caching with hash-based keys
- **Multi-Granularity**: Operations work at pixel, row, column, or block levels

### Validation Strategy
- Early validation against actual image dimensions
- Dry run mode for pipeline validation without processing
- Memory estimation before execution
- Comprehensive edge case handling (1×1 images, extreme aspect ratios, non-divisible blocks)

## Implementation Notes

When implementing new operations:
1. Inherit from BaseOperation
2. Implement all four required methods (validate, get_cache_key, estimate_memory, apply)
3. Use Pydantic models for parameter validation
4. Update ImageContext with dimension changes or warnings
5. Handle edge cases like non-divisible dimensions gracefully
6. Follow existing patterns for similar operations (e.g., Row vs Column operations mirror each other)

The codebase prioritizes clarity and safety over performance, with comprehensive validation at every step.

## Task Management Framework

Tasks are managed in `.claude/tasks/` with the following structure:
- `OVERVIEW.md`: Complete specification and design document
- `todo/`: Pending tasks to be implemented
- `wip/`: Tasks currently being worked on
- `complete/`: Finished tasks

### Workflow
1. Pick a task from `todo/` and move to `wip/`
2. Implement the task following specifications
3. Create a commit using conventional commits format
4. Move task to `complete/` when done

### Commit Convention
Use conventional commits format:
- `feat:` New feature implementation
- `fix:` Bug fixes
- `refactor:` Code restructuring
- `test:` Adding or updating tests
- `docs:` Documentation updates
- `chore:` Maintenance tasks
