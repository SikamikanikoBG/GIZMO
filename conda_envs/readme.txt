About the envs:

gizmo.yml - the conda env that is used in production

base_gizmo_config - the conda env used to build sphinx and fix dependecy issues. Till today, 27.06.2024 this is still WIP

base_gizmo_config_builds - this conda env has build flags. This may result in failure to properly build a working
                           env, but it is provided to add better env reproduction