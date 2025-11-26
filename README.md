# Compound Eye Simulator

Python + OpenGL (Compute Shaders) ray-tracing engine to simulate insect vision.

## Installation

**Prerequisites**: Python 3.10+ and a GPU supporting OpenGL 4.3+ (no macOS, sorry)
   
This project uses [uv](https://github.com/astral-sh/uv).

**Sync dependencies**:
```bash
uv sync
```

## Quick start

**`run_example.py`** is ummm, an exemple script.
It loads up the Lidar scan point clouds (add one to the `assets` folder), creates an agent with a compound eye, and runs the simulation loop.

There are other loaders for various file formats

```bash
python run_example.py
```

There are several example use cases in there, the default is the interactive mode.
Controls are displayed in the viewport.


## Project overview

*   **`graphics/`**: Core rendering logic (Raytracer, Rasterizer, GL context, shaders).
*   **`geometry/`**: Models for Compound Eyes (and basic primitives).
*   **`run_example.py`**: Main demo script demonstrating scene setup and the simulation loop.
*   **`controlled_timing_example.py`**: Skeleton script for fixed-timestep experiments.