## Dependency Installation Guide

**Our main environment and dependencies are identical to RLinf.**  
Please refer to the official documentation first:

- https://rlinf.readthedocs.io/en/latest/

The scripts below are optional helpers that follow the same embodied install flow as upstream RLinf.

We recommend using [`uv`](https://docs.astral.sh/uv/) to install Python dependencies:

```shell
pip install --upgrade uv
```

Install embodied / VLA dependencies with `requirements/install.sh`:

- Target: `embodied`
- Models: `openvla`, `openvla-oft`, `openpi`
- Envs: e.g. `maniskill_libero`, `metaworld`, `behavior`

Example (OpenVLA + ManiSkill / LIBERO):

```shell
bash requirements/install.sh embodied --model openvla --env maniskill_libero
source .venv/bin/activate
```

Override the venv directory with `--venv`:

```shell
bash requirements/install.sh embodied --model openpi --env maniskill_libero --venv openpi-venv
```
