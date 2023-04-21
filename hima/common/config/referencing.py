#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from __future__ import annotations

from pathlib import Path
from typing import Any, Type

from hima.common.config.base import TConfig, extracted, TKeyPath, read_config

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
def extracted_type_tag(config: TConfig) -> tuple[TConfig, str | None]:
    """Extracts the type tagusing the type hinting convention for configs."""
    return extracted(config, _TYPE_KEY)


def extracted_base_config(config: TConfig) -> tuple[TConfig, str | None]:
    """Extracts the base config name using the meta key convention for configs."""
    return extracted(config, _BASE_CONFIG_KEY)


class ConfigResolver:
    """
    ConfigResolver is a class that resolves the config of an object or any sub-config in general.

    In most cases object constructors expect a dictionary config with named args. For dict configs
    we support the "_base_" meta key that allows to reference another dict config to take it as
    base config and override it with the current dict. Such referencing is recursive, i.e. the base
    config may also have a "_base_" key, and so on.

    We treat "_base_" as a dot-separated key path to the base config, i.e. path containing a
    sequence of keys that are used to recursively select subconfigs from the global config.
    For example "path.to.config" means following structure:
    path:
        to:
            config: <base config>

    If the key path starts with "$" it means that the first key of the path is a name (without
    extension) of the config file to load. The tail of the path is used to select the base config
    from the loaded config. For example "$another_config.path.to.config" means the same as above,
    but the base config is loaded from the "another_config.yaml" file in the same folder as the
    current global config file.

    To prevent repeated loading of the same external config, the whole loaded config is added to
    the global config under its root key (e.g. "$another_config"), so the key path resolution
    works with only "local" key paths as is.

    The `default_base_path` is used to resolve the key path when the key path is a single key.
    In this case provided key is treated as a key in the same parent collection as the last key
    of the `default_base_path` key path. For example, if the `default_base_path` is
    "path.to.obj1_config" and the key path is "obj2_config", then it is treated as:
    path:
        to:
            obj1_config:
            obj2_config: <base config>

    In several cases config is a list (e.g. if object is a list or tuple). As they don't have
    named args, we don't support referencing with "_base_" key.

    However, for both cases we support direct referencing with a single string key path.
    """
    global_config: TConfig
    global_config_path: Path

    def __init__(self, global_config: TConfig, global_config_path: Path):
        self.global_config = global_config
        self.global_config_path = global_config_path

    def resolve(
            self, config: TConfig | str | None,
            *,
            config_type: Type[dict | list],
            config_path: str = None,
    ) -> TConfig:
        """
        Resolves the config of an object or any sub-config in general.
        See the class docstring for details.
        """
        if config is None:
            return config_type()

        if isinstance(config, str):
            # it is a reference only
            reference_path = config
            config = config_type()
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
            resolved_base_config = self.resolve(
                base_config,
                config_type=config_type,
                config_path=reference_path,
            )
            if config_type == dict:
                # by the implementation we override the innermost base config with each outer one
                resolved_base_config.update(**config)
            else:
                resolved_base_config.extend(config)

            config = resolved_base_config
        return config

    def _resolve_reference(self, config_path: str, *, default_base_path: str) -> TConfig:
        """
        Resolves the reference to the base config and returns it raw.
        See the class docstring for details.
        """
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
        """
        Loads the external config and adds it to the global config.
        See the class docstring for details.
        """
        # key: $filename_without_extension
        file_stem = reference_root_key[1:]
        config_filepath = self.global_config_path.with_stem(file_stem)

        # add it to global config by the root key, such that key_path resolution just works as is
        self.global_config[reference_root_key] = read_config(config_filepath)

    def _select_by_key_path(self, key_path: TKeyPath) -> Any:
        """
        Traverses global config and selects a sub-config by the provided key path.
        See the class docstring for details.
        """
        sub_config = self.global_config
        for key_token in key_path:
            key_token = self._parse_key_token(key_token)
            sub_config = sub_config[key_token]
        return sub_config

    @staticmethod
    def _parse_key_token(key: str) -> str | int:
        # noinspection PyShadowingNames
        def boolify(s):
            if s in ['True', 'true']:
                return True
            if s in ['False', 'false']:
                return False
            raise ValueError('Not a boolean value!')

        assert isinstance(key, str)

        # NB: try/except is widely accepted pythonic way to parse things
        # NB: order of casters is important (from most specific to most general)
        for caster in (boolify, int):
            try:
                return caster(key)
            except ValueError:
                pass
        return key
