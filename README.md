# InsectVisionSimulator

`InsectVisionSimulator` is a high-performance, biologically constrained rendering engine designed for the simulation of compound eye optics and the study of insect visual processing. 

The framework enables researchers to model species-specific ommatidial arrays, rhabdomere bundle geometries, and neural superposition wiring. It integrates photomechanical dynamics (microsaccades and retinal adaptation) into GPU-accelerated rendering pipelines, including raytracing and path tracing.

## Some features

*   **Biologically constrained eye models:** Definition of compound eyes based on empirical data, including interommatidial angles (IOA) and Snyder-based acceptance angles.
*   **Rhabdomere bundle modeling:** Support for complex rhabdomere arrangements with specific spectral sensitivities, focal plane offsets, alignment, etc.
*   **Neural superposition:** Automated wiring of lamina cartridges based on lattice-aware template snapping (snapping peripheral rhabdomere views onto neighbouring ommatidia).
*   **Photomechanical dynamics:** Simulation of rhabdomere microsaccades driven by luminance, and membrane RC integration times, etc.
*   **Hybrid rendering pipelines:** Choice of rasterization for performance or Monte-Carlo raytracing/path-tracing for physically accurate light transport and diffraction-aware sampling.

## Installation

This project utilizes `uv` for Python dependency management. To set up the environment and install the required dependencies:

```bash
# Install dependencies
uv sync

# Activate the environment
source .venv/bin/activate  # on Windows: .venv\Scripts\activate
```

*Note: A GPU supporting OpenGL 4.3+ is required for the Compute Shader-based rendering pipelines, so no macOS support, sorry*

**Troubleshoot:** If the `pytinybvh` dependency fails to install on your machine with this error: `error: external filter 'git-lfs filter-process' failed`
then you can define the `GIT_LFS_SKIP_SMUDGE` environment variable (`$env:GIT_LFS_SKIP_SMUDGE=1` on Windows, or `export GIT_LFS_SKIP_SMUDGE=1` on Linux), and then run `uv sync` again.

## Usage Examples

### 1. Initializing a Compound Eye Model
The following snippet demonstrates loading a specific species model and configuring its biological parameters.

```python
from insectvision.compound_eyes import Model
from insectvision.compound_eyes.rhabdomeres import drosophila_bundle

# Load a Drosophila model with a custom rhabdomere bundle
model = Model.from_file(
    'assets/drosophila_scaffold.npz',
    bundle=drosophila_bundle()
)

# Apply spatial scaling (conversion to meters)
model.scale(1e-6)

# Configure photomechanical gains
model.ommatidia.ampl_lat_um = 1.5  # Lateral displacement in microns
model.ommatidia.ampl_ax_um = 8.0  # Axial contraction
```

### 2. Setting up a Raytraced Scene
`InsectVisionSimulator` uses a scene-instance architecture. Assets are baked into a BVH (Bounding Volume Hierarchy) for efficient GPU intersection.

```python
from insectvision.engine import Context, Agent, Scene, Asset
from insectvision.renderers import Renderer

# The context always has to be the first thing you create
context = Context()

scene = Scene(background_color=(0.1, 0.1, 0.1))

# Load environment geometry
your_asset = Asset.from_file(name='some name', file_path='assets/some_asset.obj')
scene.add_instance(asset=your_asset, transform=(0.0, 0.0, 5.0))

# Initialise Agent (the insect)
agent = Agent(position=(0.0, 0.0, 0.0))

# Initialise the Renderer
renderer = Renderer(
    model=model, 
    scene=scene, 
    agent=agent, 
    nb_samples=128  # Monte-Carlo samples per rhabdomere
)

# [run your loop... (see examples below)]

# Always cleanup to release GPU resources and reset global context
context.free()
```

### 3. Running a simulation
In closed-loop simulations, the rendering loop updates both the physical position of the agent and the internal biological state of the sensors.

You can run in interactive mode, using a controller or mouse and keyboard inputs:
```python
# Interactive mode
while context.run_interactive(use_dashboard=True):
    context.input()     # processes inputs from keyboard / gamepad etc

    agent.translate(agent.forward * 0.5 * context.dt)
    output = renderer.step()

    context.display()      # displays to the screen
```

Headless mode:
```python
# Headless mode
all_data = []
for dt in context.run_headless(steps=1000):
    agent.translate(agent.forward * 0.5 * dt)
    output = renderer.step()
    if output: 
        all_data.append(output)

# Grab the final partial batch
final_batch = renderer.flush()
if final_batch:
    all_data.append(final_batch)
```

You can also run defer timing completely to your calling loop, either by passing the dt:
```python
dt = 1/100.0

for _ in range(BATCH_SIZE):
    agent.translate(agent.forward * 0.5 * dt)
    output = renderer.step(dt)
```

Or tick the clock explicitely:
```python
for i in range(500):

    dt = context.tick()

    agent.translate(agent.forward * 0.5 * dt)
    output = renderer.step()
```

### 4. Advanced usage

See included scripts in the `examples` folder for more examples and more functionality.