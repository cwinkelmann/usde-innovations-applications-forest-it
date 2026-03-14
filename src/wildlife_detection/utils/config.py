"""Configuration loading utilities shared across all scripts."""

from omegaconf import OmegaConf


def load_config(config_path=None, overrides=None):
    """Load a YAML config, optionally applying key=value overrides.

    Parameters
    ----------
    config_path : str or Path, optional
        Path to a YAML config file.
    overrides : list of str, optional
        List of "key=value" strings to override config values.

    Returns
    -------
    omegaconf.DictConfig
    """
    if config_path:
        cfg = OmegaConf.load(config_path)
    else:
        cfg = OmegaConf.create()

    if overrides:
        for ov in overrides:
            k, v = ov.split("=", 1)
            OmegaConf.update(cfg, k, v)

    return cfg
