#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from __future__ import annotations

from pathlib import Path
from typing import Any

from hima.common.new_config.base import TConfig, extracted, TKeyPath
from hima.common.run.entrypoint import read_config

# ==> special keys
# They are intended to be non-importable, i.e. to be used only here!


# object type alias: str
#   alias should be registered to the real type in the type registry
_TYPE_KEY = '_type_'

# name or dot-separated full path to the config entry with the base config: str
#   name
#       - name of the entry in the same parent collection;
#   collection.name | path.to.the.name
#       - fully qualified path to the entry in the config
#   $another_config_filename.path.to.the.name
#       - fully qualified path to the entry in another config file;
#           filename should be w/o extension and the file should be in the same folder
_BASE_CONFIG_KEY = '_base_'


# ==================== config meta info extraction ====================
def extracted_type(config: TConfig) -> tuple[TConfig, str | None]:
    """Extracts the type using the type hinting convention for configs."""
    return extracted(config, _TYPE_KEY)


def extracted_base_config(config: TConfig) -> tuple[TConfig, str | None]:
    """Extracts the base config name using the meta key convention for configs."""
    return extracted(config, _BASE_CONFIG_KEY)


# ==================== resolve config references ====================
class ConfigReferenceResolver:
    global_config: TConfig
    global_config_path: Path

    def __init__(self, global_config: TConfig, global_config_path: Path):
        self.global_config = global_config
        self.global_config_path = global_config_path

    def resolve_config(
            self,
            config: TConfig | list | str | None,
            *,
            object_type: type,
            config_path: str = None,
    ) -> TConfig:
        if config is None:
            return object_type()

        if isinstance(config, str):
            # it is a reference only
            reference_path = config
            config = []
        else:
            # only dicts may contain reference key in their definition;
            if isinstance(config, dict):
                config, reference_path = extracted_base_config(config)
            else:
                # others' definition is either reference-only string or just a plain value
                reference_path = None

        if reference_path is not None:
            # resolve reference first, getting an unresolved base config
            base_config = self._resolve_reference(reference_path, default_base_path=config_path)

            # recursively resolve base config the same way as current
            resolved_base_config = self.resolve_config(
                base_config,
                object_type=object_type,
                config_path=reference_path,
            )
            # by the implementation we override the innermost base config with each outer one
            resolved_base_config.update(**config)
            config = resolved_base_config

        return config

    def _resolve_reference(self, config_path: str, *, default_base_path: str) -> TConfig:
        key_path: TKeyPath = config_path.split('.')

        if key_path[0].startswith('$') and key_path[0] not in self.global_config:
            self._load_external_config(key_path[0])

        elif len(key_path) == 1:
            # single key means a reference to file neighboring the `default_base_path` file
            key = key_path[0]
            # hence, replace the last key of the `default_base_path` key path
            key_path = default_base_path.split('.')
            key_path[-1] = key

        return self._select_by_key_path(key_path)

    def _load_external_config(self, reference_root_key: str):
        # key: $filename_without_extension
        file_stem = reference_root_key[1:]
        config_filepath = self.global_config_path.with_stem(file_stem)

        # add it to global config by the root key, such that key_path resolution just works as is
        self.global_config[reference_root_key] = read_config(config_filepath)

    def _select_by_key_path(self, key_path: TKeyPath) -> Any:
        sub_config = self.global_config
        for key_token in key_path:
            sub_config = sub_config[key_token]
        return sub_config
