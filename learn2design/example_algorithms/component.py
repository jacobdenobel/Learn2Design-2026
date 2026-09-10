"""Component-wise learned optimizer for Learn2Design / dfbench.

This file follows the style of the optimizers in ``dfbench.algorithms.gradient_based``.

The UIFO problem exposes a flat continuous parameter vector, but
``Objective.optimization_pairs`` maps every coordinate back to a physical
component/property. This optimizer groups coordinates into reusable families:

    mirror, beamsplitter, squeezer, laser, space

Each family shares a tiny MLP. The MLP sees the current unbounded parameters,
gradient, Adam moments/direction, and a small topology/context vector, and returns:

    dz = -lr * exp(log_scale) * adam_direction
         + residual_scale * tanh(residual)

The final layer is initialized to zero, so with no pretrained policy this is
exactly Adam. Later, the same file can load pretrained component policies from
an NPZ checkpoint produced from the supplied dataset.

Optionally, ``online_policy_learning_rate > 0`` enables a cheap experimental
first-order online adaptation of the component MLPs without Hessians.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

try:
    import optax
except ImportError as exc:
    raise ImportError(
        "optax is required. Install with: uv add 'dfbench[optax]'"
    ) from exc

from jaxtyping import Array, Float

from dfbench.core.algorithm import AlgorithmType, OptimizationAlgorithm
from dfbench.core.objective import Objective


PROPERTY_NAMES = (
    "reflectivity",
    "tuning",
    "mass",
    "length",
    "power",
    "db",
    "angle",
)
PROPERTY_TO_SLOT = {name: i for i, name in enumerate(PROPERTY_NAMES)}
N_PROPERTY_SLOTS = len(PROPERTY_NAMES)

FAMILIES = (
    "mirror",
    "beamsplitter",
    "squeezer",
    "laser",
    "space",
)

# row, col, boundary,
# own center type (2),
# center orientation (4),
# N/S/W/E neighbor center types (4),
# global boundary fractions (4),
# role (7)
CONTEXT_DIM = 24

# Per property slot:
# z, signed-log grad, m_hat, Adam direction, RMS grad, mask
N_DYNAMIC_FEATURES_PER_SLOT = 6
INPUT_DIM = N_DYNAMIC_FEATURES_PER_SLOT * N_PROPERTY_SLOTS + CONTEXT_DIM + 1


@dataclass(frozen=True)
class _Group:
    family: str
    name: str
    items: tuple[tuple[int, str], ...]
    targets: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class _FamilyLayout:
    gather_indices: jax.Array
    mask: jax.Array
    context: jax.Array
    rows: jax.Array
    cols: jax.Array
    flat_indices: jax.Array


def _flatten_pair_targets(value: Any) -> list[tuple[str, str]]:
    """Return all (component, property) leaves from a normal/coupled pair."""
    result: list[tuple[str, str]] = []

    def visit(x: Any) -> None:
        if (
            isinstance(x, (list, tuple))
            and len(x) == 2
            and isinstance(x[0], str)
            and isinstance(x[1], str)
        ):
            result.append((x[0], x[1]))
            return

        if isinstance(x, (list, tuple)):
            for y in x:
                visit(y)

    visit(value)
    return result


def _family_and_name(
    parameter_index: int,
    component: str,
    prop: str,
) -> tuple[str, str]:
    if prop == "length":
        return "space", f"space-{parameter_index}"

    if component.endswith("sus"):
        base = component[:-3]
        if base.startswith("center"):
            return "beamsplitter", base
        return "mirror", base

    if component.startswith("center"):
        return "beamsplitter", component

    if "bhbs" in component:
        return "beamsplitter", component

    if re.fullmatch(r"m\d\d", component):
        return "mirror", component

    if component.startswith(("ml", "mr", "mt", "mb")):
        return "mirror", component

    if component.startswith("boundary"):
        if prop == "power":
            return "laser", component
        if prop in {"db", "angle"}:
            return "squeezer", component

    raise ValueError(
        f"Cannot classify ({component!r}, {prop!r}). "
        "Inspect objective.optimization_pairs and extend _family_and_name()."
    )


def _build_groups(optimization_pairs: list[Any]) -> list[_Group]:
    grouped: dict[tuple[str, str], dict[str, Any]] = {}

    for i, pair in enumerate(optimization_pairs):
        targets = _flatten_pair_targets(pair)
        if not targets:
            raise ValueError(f"Could not parse optimization pair {i}: {pair!r}")

        prop = targets[0][1]
        if prop not in PROPERTY_TO_SLOT:
            raise ValueError(f"Unsupported optimized property: {prop!r}")

        family, name = _family_and_name(i, targets[0][0], prop)
        key = (family, name)

        if key not in grouped:
            grouped[key] = {"items": [], "targets": []}

        grouped[key]["items"].append((i, prop))
        grouped[key]["targets"].extend(targets)

    return [
        _Group(
            family=family,
            name=name,
            items=tuple(data["items"]),
            targets=tuple(data["targets"]),
        )
        for (family, name), data in grouped.items()
    ]


_CENTER_CODES = {
    "A": ("beamsplitter", "left"),
    "B": ("beamsplitter", "right"),
    "C": ("beamsplitter", "top"),
    "D": ("beamsplitter", "bottom"),
    "E": ("directional_beamsplitter", "left"),
    "F": ("directional_beamsplitter", "right"),
    "G": ("directional_beamsplitter", "top"),
    "H": ("directional_beamsplitter", "bottom"),
}
_BOUNDARY_CODES = {
    "L": "laser",
    "S": "squeezer",
    "D": "detector",
    "H": "balanced_homodyne",
}


def _find_in_nested_dict(d: Any, key: str) -> Any | None:
    if not isinstance(d, dict):
        return None
    if key in d:
        return d[key]
    for value in d.values():
        if isinstance(value, dict):
            found = _find_in_nested_dict(value, key)
            if found is not None:
                return found
    return None


def _boundary_positions(size: int) -> list[str]:
    grid = size + 2
    result = [f"0{c}" for c in range(1, grid - 1)]
    for r in range(1, grid - 1):
        result.extend([f"{r}0", f"{r}{grid - 1}"])
    result.extend([f"{grid - 1}{c}" for c in range(1, grid - 1)])
    return result


def _decode_topology(
    topology: str | None,
    size: int,
) -> tuple[dict[str, tuple[str, str]], dict[str, str]]:
    if topology is None:
        return {}, {}

    interior, boundary = topology.split("-")
    ipos = [f"{r}{c}" for r in range(1, size + 1) for c in range(1, size + 1)]
    bpos = _boundary_positions(size)

    if len(interior) != len(ipos) or len(boundary) != len(bpos):
        return {}, {}

    centers = {
        pos: _CENTER_CODES[ch]
        for pos, ch in zip(ipos, interior, strict=True)
    }
    boundaries = {
        pos: _BOUNDARY_CODES[ch]
        for pos, ch in zip(bpos, boundary, strict=True)
    }
    return centers, boundaries


def _extract_coordinates(text: str, size: int) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    maximum = size + 1

    for match in re.finditer(r"(?<!\d)(\d)(\d)(?!\d)", text):
        r, c = int(match.group(1)), int(match.group(2))
        if 0 <= r <= maximum and 0 <= c <= maximum:
            coords.append((r, c))

    return coords


def _nearest_interior_cell(
    coords: list[tuple[int, int]],
    size: int,
) -> tuple[int, int] | None:
    if not coords:
        return None
    r, c = coords[0]
    return min(max(r, 1), size), min(max(c, 1), size)


def _component_role(name: str, family: str) -> np.ndarray:
    # left, right, top, bottom, boundary, center, space
    role = np.zeros(7, dtype=np.float32)

    if family == "space":
        role[6] = 1.0
    elif name.startswith("ml"):
        role[0] = 1.0
    elif name.startswith("mr"):
        role[1] = 1.0
    elif name.startswith("mt"):
        role[2] = 1.0
    elif name.startswith("mb"):
        role[3] = 1.0
    elif name.startswith("m") or name.startswith("boundary"):
        role[4] = 1.0
    elif name.startswith("center"):
        role[5] = 1.0

    return role


def _group_context(
    group: _Group,
    *,
    centers: dict[str, tuple[str, str]],
    boundaries: dict[str, str],
    size: int,
) -> np.ndarray:
    all_coords: list[tuple[int, int]] = []

    for component, _ in group.targets:
        all_coords.extend(_extract_coordinates(component, size))

    if not all_coords:
        all_coords.extend(_extract_coordinates(group.name, size))

    if all_coords:
        rr = np.mean([r for r, _ in all_coords]) / max(size + 1, 1)
        cc = np.mean([c for _, c in all_coords]) / max(size + 1, 1)
        boundary_flag = float(
            any(r in {0, size + 1} or c in {0, size + 1} for r, c in all_coords)
        )
    else:
        rr = cc = boundary_flag = 0.0

    cell = _nearest_interior_cell(all_coords, size)

    own_type = np.zeros(2, dtype=np.float32)
    orientation = np.zeros(4, dtype=np.float32)
    neighbour_types = np.zeros(4, dtype=np.float32)

    orient_index = {"left": 0, "right": 1, "top": 2, "bottom": 3}

    if cell is not None and centers:
        r, c = cell
        center = centers.get(f"{r}{c}")

        if center is not None:
            comp, orient = center
            own_type[0 if comp == "beamsplitter" else 1] = 1.0
            orientation[orient_index[orient]] = 1.0

        for k, (dr, dc) in enumerate(((-1, 0), (1, 0), (0, -1), (0, 1))):
            nr, nc = r + dr, c + dc
            if 1 <= nr <= size and 1 <= nc <= size:
                neighbour = centers.get(f"{nr}{nc}")
                if neighbour is not None:
                    neighbour_types[k] = (
                        1.0 if neighbour[0] == "beamsplitter" else -1.0
                    )

    boundary_fraction = np.zeros(4, dtype=np.float32)
    if boundaries:
        denom = max(len(boundaries), 1)
        order = ("laser", "squeezer", "detector", "balanced_homodyne")
        for i, kind in enumerate(order):
            boundary_fraction[i] = sum(v == kind for v in boundaries.values()) / denom

    role = _component_role(group.name, group.family)

    context = np.concatenate(
        [
            np.asarray([rr, cc, boundary_flag], dtype=np.float32),
            own_type,
            orientation,
            neighbour_types,
            boundary_fraction,
            role,
        ]
    )

    assert context.shape == (CONTEXT_DIM,)
    return context


def _build_layouts(
    objective: Objective,
    groups: list[_Group],
) -> dict[str, _FamilyLayout]:
    n_params = len(objective.optimization_pairs)

    spec = objective.problem_spec
    topology = _find_in_nested_dict(spec, "topology")
    size_value = _find_in_nested_dict(spec, "size")
    size = int(size_value) if size_value is not None else 3

    centers, boundaries = _decode_topology(topology, size)

    result: dict[str, _FamilyLayout] = {}

    for family in FAMILIES:
        family_groups = [g for g in groups if g.family == family]
        if not family_groups:
            continue

        idx = np.full(
            (len(family_groups), N_PROPERTY_SLOTS),
            n_params,
            dtype=np.int32,
        )
        mask = np.zeros_like(idx, dtype=np.float32)
        context = np.zeros((len(family_groups), CONTEXT_DIM), dtype=np.float32)

        for row, group in enumerate(family_groups):
            for parameter_index, prop in group.items:
                col = PROPERTY_TO_SLOT[prop]
                if mask[row, col]:
                    raise ValueError(
                        f"Duplicate property {prop!r} in physical group {group.name!r}"
                    )
                idx[row, col] = parameter_index
                mask[row, col] = 1.0

            context[row] = _group_context(
                group,
                centers=centers,
                boundaries=boundaries,
                size=size,
            )

        rows, cols = np.nonzero(mask)
        flat_indices = idx[rows, cols]

        result[family] = _FamilyLayout(
            gather_indices=jnp.asarray(idx),
            mask=jnp.asarray(mask),
            context=jnp.asarray(context),
            rows=jnp.asarray(rows),
            cols=jnp.asarray(cols),
            flat_indices=jnp.asarray(flat_indices),
        )

    return result


def _init_mlp(
    key: jax.Array,
    *,
    hidden_size: int,
) -> dict[str, jax.Array]:
    k1, k2 = jax.random.split(key)

    w1 = (
        jax.random.normal(k1, (INPUT_DIM, hidden_size))
        / jnp.sqrt(float(INPUT_DIM))
    )
    b1 = jnp.zeros(hidden_size)

    w2 = (
        jax.random.normal(k2, (hidden_size, hidden_size))
        / jnp.sqrt(float(hidden_size))
    )
    b2 = jnp.zeros(hidden_size)

    # Zero head => log_scale=0, residual=0 => exact Adam.
    w3 = jnp.zeros((hidden_size, 2 * N_PROPERTY_SLOTS))
    b3 = jnp.zeros(2 * N_PROPERTY_SLOTS)

    return {
        "w1": w1,
        "b1": b1,
        "w2": w2,
        "b2": b2,
        "w3": w3,
        "b3": b3,
    }


@jax.jit
def _mlp_apply(
    params: dict[str, jax.Array],
    x: jax.Array,
) -> jax.Array:
    h = jax.nn.gelu(x @ params["w1"] + params["b1"])
    h = jax.nn.gelu(h @ params["w2"] + params["b2"])
    return h @ params["w3"] + params["b3"]


def _initialize_policy(
    key: jax.Array,
    *,
    hidden_size: int,
) -> dict[str, dict[str, jax.Array]]:
    keys = jax.random.split(key, len(FAMILIES))
    return {
        family: _init_mlp(k, hidden_size=hidden_size)
        for family, k in zip(FAMILIES, keys, strict=True)
    }


def _load_policy_npz(
    path: str | Path,
    policy: dict[str, dict[str, jax.Array]],
) -> dict[str, dict[str, jax.Array]]:
    path = Path(path)

    with np.load(path) as data:
        loaded: dict[str, dict[str, jax.Array]] = {}

        for family in FAMILIES:
            family_params = {}

            for name in ("w1", "b1", "w2", "b2", "w3", "b3"):
                key = f"{family}_{name}"

                if key not in data:
                    raise KeyError(f"Missing {key!r} in {path}")

                value = jnp.asarray(data[key])

                if value.shape != policy[family][name].shape:
                    raise ValueError(
                        f"{key}: got shape {value.shape}, "
                        f"expected {policy[family][name].shape}"
                    )

                family_params[name] = value

            loaded[family] = family_params

    return loaded


def _signed_log(x: jax.Array) -> jax.Array:
    return jnp.sign(x) * jnp.log1p(jnp.abs(x))


def _policy_delta(
    policy: dict[str, dict[str, jax.Array]],
    layouts: dict[str, _FamilyLayout],
    *,
    z: jax.Array,
    grad: jax.Array,
    m_hat: jax.Array,
    v_hat: jax.Array,
    learning_rate: float,
    residual_scale: float,
    epsilon: float,
    progress: float | jax.Array,
) -> jax.Array:
    n_params = z.shape[0]

    zero = jnp.zeros((1,), dtype=z.dtype)
    z_pad = jnp.concatenate([z, zero])
    g_pad = jnp.concatenate([grad, zero])
    m_pad = jnp.concatenate([m_hat, zero])
    v_pad = jnp.concatenate([v_hat, zero])

    delta = jnp.zeros_like(z)

    for family, layout in layouts.items():
        index = layout.gather_indices
        mask = layout.mask

        z_slot = z_pad[index]
        g_slot = g_pad[index]
        m_slot = m_pad[index]
        v_slot = v_pad[index]

        rms = jnp.sqrt(jnp.maximum(v_slot, 0.0))
        adam_direction = m_slot / (rms + epsilon)

        features = jnp.concatenate(
            [
                jnp.tanh(z_slot / 5.0),
                jnp.tanh(_signed_log(g_slot)),
                jnp.tanh(m_slot),
                jnp.tanh(adam_direction),
                jnp.tanh(jnp.log1p(rms)),
                mask,
                layout.context,
                jnp.full((index.shape[0], 1), progress, dtype=z.dtype),
            ],
            axis=1,
        )

        output = _mlp_apply(policy[family], features)

        log_scale = jnp.clip(
            output[:, :N_PROPERTY_SLOTS],
            -2.5,
            2.5,
        )
        residual = jnp.tanh(output[:, N_PROPERTY_SLOTS:])

        dz_slot = (
            -learning_rate * jnp.exp(log_scale) * adam_direction
            + residual_scale * residual
        )
        dz_slot = dz_slot * mask

        values = dz_slot[layout.rows, layout.cols]
        delta = delta.at[layout.flat_indices].set(values)

    return delta


class ComponentPolicyAdam(OptimizationAlgorithm):
    """Adam augmented with reusable learned component-specific update policies."""

    algorithm_str = "component_policy_adam"
    algorithm_type = AlgorithmType.GRADIENT_BASED

    def __init__(self) -> None:
        pass

    def optimize(
        self,
        objective: Objective,
        init_params: Float[Array, "..."] | None = None,
        random_seed: int | None = None,
        learning_rate: float = 0.1,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        hidden_size: int = 64,
        residual_scale: float = 0.02,
        policy_path: str | None = None,
        online_policy_learning_rate: float = 0.0,
        policy_weight_decay: float = 0.0,
        patience: int | None = None,
        restart_every: int | None = None,
        restart_noise_std: float = 0.15,
        **_: Any,
    ) -> None:
        obj = objective

        _, rng_key = self.prepare(
            obj,
            unbounded=True,
            random_seed=random_seed,
        )

        params = (
            obj.random_params_unbounded()
            if init_params is None
            else jnp.asarray(init_params)
        )

        groups = _build_groups(obj.optimization_pairs)
        layouts = _build_layouts(obj, groups)

        rng_key, policy_key = jax.random.split(rng_key)
        policy = _initialize_policy(
            policy_key,
            hidden_size=hidden_size,
        )

        if policy_path is not None:
            policy = _load_policy_npz(policy_path, policy)

        if online_policy_learning_rate > 0.0:
            policy_optimizer = optax.adamw(
                learning_rate=online_policy_learning_rate,
                weight_decay=policy_weight_decay,
            )
            policy_opt_state = policy_optimizer.init(policy)
        else:
            policy_optimizer = None
            policy_opt_state = None

        m = jnp.zeros_like(params)
        v = jnp.zeros_like(params)

        previous_policy_inputs = None

        obj.warmup_value_and_grad()
        obj.start_logging()

        step = 0

        while not obj.budget_exceeded:
            _, grad = obj.value_and_grad(params)

            # Experimental online first-order hypergradient for the previous update.
            if (
                policy_optimizer is not None
                and previous_policy_inputs is not None
            ):
                prev_z, prev_g, prev_mh, prev_vh, prev_progress = previous_policy_inputs
                grad_stop = jax.lax.stop_gradient(grad)

                def surrogate_loss(policy_params):
                    previous_delta = _policy_delta(
                        policy_params,
                        layouts,
                        z=prev_z,
                        grad=prev_g,
                        m_hat=prev_mh,
                        v_hat=prev_vh,
                        learning_rate=learning_rate,
                        residual_scale=residual_scale,
                        epsilon=epsilon,
                        progress=prev_progress,
                    )
                    return jnp.vdot(grad_stop, previous_delta)

                policy_grad = jax.grad(surrogate_loss)(policy)

                updates, policy_opt_state = policy_optimizer.update(
                    policy_grad,
                    policy_opt_state,
                    policy,
                )
                policy = optax.apply_updates(policy, updates)

            if patience is not None and obj.evals_since_improvement > patience:
                break

            step += 1

            m = beta1 * m + (1.0 - beta1) * grad
            v = beta2 * v + (1.0 - beta2) * (grad * grad)

            m_hat = m / (1.0 - beta1**step)
            v_hat = v / (1.0 - beta2**step)

            progress = float(obj.budget_progress_fraction)

            delta = _policy_delta(
                policy,
                layouts,
                z=params,
                grad=grad,
                m_hat=m_hat,
                v_hat=v_hat,
                learning_rate=learning_rate,
                residual_scale=residual_scale,
                epsilon=epsilon,
                progress=progress,
            )

            previous_policy_inputs = (
                params,
                grad,
                m_hat,
                v_hat,
                progress,
            )

            params = params + delta

            if (
                restart_every is not None
                and restart_every > 0
                and step % restart_every == 0
                and not obj.budget_exceeded
            ):
                base = (
                    jnp.asarray(obj.best_params)
                    if obj.best_params is not None
                    else obj.random_params_unbounded()
                )

                rng_key, noise_key = jax.random.split(rng_key)
                params = base + restart_noise_std * jax.random.normal(
                    noise_key,
                    shape=base.shape,
                )

                m = jnp.zeros_like(params)
                v = jnp.zeros_like(params)
                previous_policy_inputs = None

