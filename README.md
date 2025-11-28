## Quick Installation

Install **PyTorch** first following the [official instructions](https://pytorch.org/get-started/locally/).

Then, install Genesis via PyPI:
```bash
pip install genesis-world  # Requires Python>=3.10,<3.14;
```

Then clone the repository and install locally:
```bash
git clone https://github.com/andrevic1/Genesis.git
cd Genesis
pip install -e ".[dev]"
```
It is recommended to systematically execute `pip install -e ".[dev]"` after moving HEAD to make sure that all dependencies and entrypoints are up-to-date.

## Documentation

Comprehensive documentation is available in [English](https://genesis-world.readthedocs.io/en/latest/user_guide/index.html), [Chinese](https://genesis-world.readthedocs.io/zh-cn/latest/user_guide/index.html), and [Japanese](https://genesis-world.readthedocs.io/ja/latest/user_guide/index.html). This includes detailed installation steps, tutorials, and API references.

## License and Acknowledgments

The Genesis source code is licensed under Apache 2.0.

Genesis's development has been made possible thanks to these open-source projects:

- [Taichi](https://github.com/taichi-dev/taichi): High-performance cross-platform compute backend. Kudos to the Taichi team for their technical support!
- [FluidLab](https://github.com/zhouxian/FluidLab): Reference MPM solver implementation.
- [SPH_Taichi](https://github.com/erizmr/SPH_Taichi): Reference SPH solver implementation.
- [Ten Minute Physics](https://matthias-research.github.io/pages/tenMinutePhysics/index.html) and [PBF3D](https://github.com/WASD4959/PBF3D): Reference PBD solver implementations.
- [MuJoCo](https://github.com/google-deepmind/mujoco): Reference for rigid body dynamics.
- [libccd](https://github.com/danfis/libccd): Reference for collision detection.
- [PyRender](https://github.com/mmatl/pyrender): Rasterization-based renderer.
- [LuisaCompute](https://github.com/LuisaGroup/LuisaCompute) and [LuisaRender](https://github.com/LuisaGroup/LuisaRender): Ray-tracing DSL.
- [Madrona](https://github.com/shacklettbp/madrona) and [Madrona-mjx](https://github.com/shacklettbp/madrona_mjx): Batch renderer backend
