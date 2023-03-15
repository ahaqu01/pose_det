import importlib


def get_cfg(config_name):
    config_name = config_name.split(".")[0].split("/")[-1]
    module_name = "." + config_name
    package_name = "action_det.src.configs"
    cfg = importlib.import_module(module_name, package=package_name)
    return cfg
