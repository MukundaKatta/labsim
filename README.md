# LABSIM - Virtual STEM Lab

Simulate physics, chemistry, and biology experiments from the command line.

## Installation

```bash
pip install -e ".[dev]"
```

## Usage

```bash
# Run a predefined experiment
labsim run projectile --angle 45 --velocity 20

# Interactive experiment mode
labsim experiment physics pendulum --length 1.5 --angle 30

# Generate a lab report
labsim report projectile --format rich
```

## Labs

- **Physics**: Projectile motion, simple pendulum, Ohm's law circuits, thin-lens optics
- **Chemistry**: Ideal gas law, reaction kinetics, acid-base titration, molecular modeling
- **Biology**: Cell division (logistic growth), Mendelian genetics, population dynamics (Lotka-Volterra)

## Examples

```python
from labsim.labs.physics import PhysicsLab
from labsim.simulator.engine import SimulationEngine

lab = PhysicsLab()
params = lab.projectile_motion(angle_deg=45, velocity=25)
engine = SimulationEngine()
result = engine.solve(params)
```

## Author

Mukunda Katta
