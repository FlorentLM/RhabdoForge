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

## Usage Examples

### 1. Initializing a Compound Eye Model
The following snippet demonstrates loading a specific species model and configuring its biological parameters.

```python
from insectvision.compound_eyes import CompoundEyeModel
from insectvision.compound_eyes.kernel import drosophila_kernel

# Load a Drosophila model with a custom rhabdomere kernel
model = CompoundEyeModel.from_file(
    'species_models/drosophila_custom.npz', 
    kernel=drosophila_kernel()
)

# Apply spatial scaling (conversion to meters)
model.scale(1e-6)

# Configure photomechanical gains
with model.unlock(lenses=True):
    model.lenses.gain_lat_um = 1.5  # Lateral displacement in microns
    model.lenses.gain_ax_um = 8.0   # Axial contraction
```

### 2. Setting up a Raytraced Scene
`InsectVisionSimulator` uses a scene-instance architecture. Assets are baked into a BVH (Bounding Volume Hierarchy) for efficient GPU intersection.

```python
from insectvision.engine import Context, Agent, Scene, Asset
from insectvision.renderers import Raytracer

context = Context()
scene = Scene(background_color=(0.1, 0.1, 0.1))

# Load environment geometry
your_asset = Asset.from_file(name='some name', file_path='assets/some_asset.obj')
scene.add_instance(asset=your_asset, transform=(0.0, 0.0, 5.0))

# Initialize Agent (the insect)
agent = Agent(position=(0.0, 0.0, 0.0))

# Initialize the Renderer
renderer = Raytracer(
    model=model, 
    scene=scene, 
    agent=agent, 
    context=context,
    nb_samples=128  # Monte-Carlo samples per receptor
)
```

### 3. Running a Biological Simulation Step
In closed-loop simulations, the rendering loop updates both the physical position of the agent and the internal biological state of the sensors.

```python
# Use a fixed biological time step
context.time_step = 1/1000.0  # 1 ms resolution

while context.run_interactive(agent=agent, scene=scene, renderer=renderer):
    # Update agent state
    agent.translate(agent.forward * 1.0 * context.dt)
    
    # Compute one biological step
    # This processes optics, rhabdomere dynamics, and adaptation
    visual_output = renderer.step()
    
    # visual_output.per_cartridge provides the neural-superposition signal
    # visual_output.per_lens provides the physical ommatidial signal
    l2_signals = visual_output.per_cartridge[:, :6, 3] # R1-R6 radiance
    
    context.draw()
```

### 4. Advanced usage

See included scripts in the `examples` folder for more examples and more functionality.