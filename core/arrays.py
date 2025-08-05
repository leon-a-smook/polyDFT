# core/arrays.py
import jax
import jax.numpy as jnp

Array = jax.Array  # For hint types
xp = jnp  # Drop-in replacement for NumPy API

def get_device_info():
    dev = jax.devices()[0]
    return {
        "device": dev.device_kind,
        "platform": dev.platform,
        "architecture": dev.device_kind,
    }

def to_numpy(array):
    return jax.device_get(array)