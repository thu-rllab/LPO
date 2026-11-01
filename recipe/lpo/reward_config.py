from copy import deepcopy


_REWARD_PATHS = {
    "countdown": "./recipe/lpo/reward_func/countdown_reward.py",
    "math": "./recipe/lpo/reward_func/math_reward.py",
    "geo3k": "./recipe/lpo/reward_func/geo3k_reward.py",
}
_COUNTDOWN_NOISE_PATH = "./recipe/lpo/reward_func/countdown_reward_noise.py"


def _contains_dataset(data_files, dataset):
    if isinstance(data_files, str):
        return dataset in data_files
    return any(dataset in str(path) for path in data_files)


def _build_reward_function_config(
    data_files, custom_reward_function, *, training
):
    configured = deepcopy(dict(custom_reward_function or {}))
    reward_kwargs = dict(configured.get("reward_kwargs") or {})

    train_reward_kwargs = {}
    noise_ratio = None
    if training:
        train_reward_kwargs = dict(configured.get("train_reward_kwargs") or {})
        noise_ratio = train_reward_kwargs.pop("noise_ratio", None)
        reward_kwargs.update(train_reward_kwargs)

    for dataset, reward_path in _REWARD_PATHS.items():
        if _contains_dataset(data_files, dataset):
            configured["path"] = reward_path
            break

    if training and _contains_dataset(data_files, "countdown"):
        if noise_ratio is not None and not 0.0 <= noise_ratio <= 1.0:
            raise ValueError(
                f"noise_ratio must be between 0 and 1, got {noise_ratio}"
            )
        if noise_ratio:
            configured["path"] = _COUNTDOWN_NOISE_PATH
            reward_kwargs["noise_ratio"] = noise_ratio

    configured["name"] = "compute_score"
    configured["reward_kwargs"] = reward_kwargs
    return configured


def build_reward_function_configs(
    train_files, val_files, custom_reward_function
):
    """Build isolated custom reward settings for training and validation."""
    train_config = _build_reward_function_config(
        train_files,
        custom_reward_function,
        training=True,
    )
    val_config = _build_reward_function_config(
        val_files,
        custom_reward_function,
        training=False,
    )
    return train_config, val_config
