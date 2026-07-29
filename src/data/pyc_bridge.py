#!/usr/bin/env python
"""Load the research feature extractors from surviving bytecode.

The `.py` sources under `src/data/` were deleted and had never been committed;
only `__pycache__/*.cpython-311.pyc` remains. Python can import those directly
via SourcelessFileLoader, which recovers the full 272-feature research pipeline
without a decompiler.

This is a *research-only* path. It requires Python 3.11 (the bytecode magic is
version-locked) and the `pn26` env. Bytecode must never ship in a submission:
the evaluation container is Python 3.10, and shipping compiled artifacts would
undermine the open-source requirement. Clean source for the features that
actually ship gets written separately and validated against these modules as
the oracle.

Usage:
    from src.data.pyc_bridge import load_feature_modules
    mods = load_feature_modules()
    feats = mods['features'].extract_all_features(...)
"""

import importlib.machinery
import importlib.util
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.dirname(_HERE)
_REPO_ROOT = os.path.dirname(_SRC)
_CACHE_DIR = os.path.join(_HERE, '__pycache__')

# `features.py` is an orchestrator that imports the others as `data.<name>`, so
# the submodules must be registered under both their bare name and that prefix
# before it is loaded. Order matters: dependencies first.
MODULE_ORDER = [
    'edf_reader',
    'features_caisr',
    'features_cardio',
    'features_eeg',
    'harmonization',
    'site_filter',
    'features',
]

TRAINING_MODULES = ['stacking']


def _pyc_path(name, subdir=None):
    base = os.path.join(_SRC, subdir, '__pycache__') if subdir else _CACHE_DIR
    return os.path.join(base, f'{name}.cpython-311.pyc')


def _install_helper_shims():
    """Restore helper_code names the bytecode was compiled against.

    The research modules predate the July upstream, which renamed
    get_standardized_race/ethnicity to load_race/load_ethnicity. helper_code.py
    is vendored verbatim and must not be edited, so the aliases are injected
    into its namespace at import time instead. This affects the research path
    only; team_code.py calls load_race directly.
    """
    import helper_code

    aliases = {
        'get_standardized_race': 'load_race',
        'get_standardized_ethnicity': 'load_ethnicity',
    }
    for old, new in aliases.items():
        if not hasattr(helper_code, old) and hasattr(helper_code, new):
            setattr(helper_code, old, getattr(helper_code, new))


def _install_specparam_shims():
    """Expose the specparam 1.x attribute names on the 2.x model class.

    extract_spectral_shape_features reads `aperiodic_params_` and
    `peak_params_`, which existed in fooof and specparam 1.x. The installed
    specparam is 2.0.0rc6, where those became `get_params('aperiodic')` and
    `get_params('peak')`. The extractor wraps its fit in a broad `except
    Exception`, so the AttributeError was swallowed and all six spectral-shape
    features came back NaN on every one of 1,103 records: a silent failure that
    looked exactly like a data property.

    Restoring the old names as properties is preferable to editing the bytecode
    and keeps the recovered extractors byte-identical to what produced the
    cached v6 matrix.
    """
    try:
        from specparam import SpectralModel
    except ImportError:
        return

    def _params(model, kind):
        try:
            return model.get_params(kind)
        except Exception:  # noqa: BLE001 - unfit model or no peaks found
            return None

    if not hasattr(SpectralModel, 'aperiodic_params_'):
        SpectralModel.aperiodic_params_ = property(
            lambda self: _params(self, 'aperiodic'))
    if not hasattr(SpectralModel, 'peak_params_'):
        SpectralModel.peak_params_ = property(
            lambda self: _params(self, 'peak'))


def _load_one(name, path):
    loader = importlib.machinery.SourcelessFileLoader(name, path)
    spec = importlib.util.spec_from_loader(name, loader)
    module = importlib.util.module_from_spec(spec)
    # Register before exec so circular/self references resolve.
    sys.modules[name] = module
    sys.modules[f'data.{name}'] = module
    loader.exec_module(module)
    return module


def load_feature_modules(include_training=False):
    """Import every recoverable research module. Returns {name: module}.

    Raises RuntimeError on Python != 3.11, where the bytecode will not load.
    """
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError(
            f'pyc_bridge needs Python 3.11 (bytecode is version-locked); '
            f'got {sys.version_info.major}.{sys.version_info.minor}. '
            f'Use the pn26 env.'
        )

    if _SRC not in sys.path:
        sys.path.insert(0, _SRC)
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)

    _install_helper_shims()
    _install_specparam_shims()

    # `features.py` does `from data.features_caisr import ...`, so a stub
    # package named `data` must exist for those imports to resolve.
    if 'data' not in sys.modules:
        pkg = importlib.util.module_from_spec(
            importlib.util.spec_from_loader('data', loader=None, is_package=True))
        pkg.__path__ = [_HERE]
        sys.modules['data'] = pkg

    loaded = {}
    for name in MODULE_ORDER:
        path = _pyc_path(name)
        if not os.path.exists(path):
            continue
        loaded[name] = _load_one(name, path)

    if include_training:
        for name in TRAINING_MODULES:
            path = _pyc_path(name, subdir='training')
            if os.path.exists(path):
                loaded[name] = _load_one(name, path)

    if not loaded:
        raise RuntimeError(f'No bytecode modules found under {_CACHE_DIR}')

    return loaded


def feature_api():
    """Return the callables needed to extract the full research feature set."""
    mods = load_feature_modules()
    api = {}
    for mod in mods.values():
        for attr in dir(mod):
            if attr.startswith(('extract_', 'precompute_', 'load_', 'standardize_',
                                'get_', 'compute_', 'filter_', 'greedy_')):
                api.setdefault(attr, getattr(mod, attr))
    return api


if __name__ == '__main__':
    mods = load_feature_modules(include_training=True)
    print(f'Loaded {len(mods)} modules from bytecode:')
    for name, mod in sorted(mods.items()):
        public = [a for a in dir(mod)
                  if not a.startswith('_') and callable(getattr(mod, a))]
        print(f'  {name:20s} {len(public):3d} callables')
    api = feature_api()
    extractors = sorted(a for a in api if a.startswith(('extract_', 'precompute_')))
    print(f'\n{len(extractors)} extractors recovered:')
    for e in extractors:
        print(f'  {e}')
