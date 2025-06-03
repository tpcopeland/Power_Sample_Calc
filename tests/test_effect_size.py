"""Unit tests for :func:`calculate_effect_size`.

The main module depends on heavy packages like ``numpy`` and ``statsmodels``
that are not available in the execution environment used for testing.  The
``import_power_sample_calc`` helper creates lightweight stub modules when these
dependencies cannot be imported and then loads ``power_sample_calc.py``.  This
allows the effect size calculations to be tested without installing the full
stack.
"""

import importlib
import importlib.util
import sys
import types
import math
from pathlib import Path
import pytest


def import_power_sample_calc():
    """Load ``power_sample_calc`` with stubbed dependencies.

    Each required third-party package is imported if available.  When a module
    is missing, a small stub with just the attributes needed for the tests is
    inserted into ``sys.modules``.  This keeps the loader simple while avoiding
    heavyweight installs.
    """
    # Ensure required modules exist, creating minimal stubs when missing
    def ensure_module(name, module):
        if name not in sys.modules:
            sys.modules[name] = module

    try:
        import streamlit  # type: ignore
    except Exception:
        class DummyModule(types.ModuleType):
            def __getattr__(self, name):
                if name == 'expander':
                    class Dummy:
                        def __enter__(self_inner):
                            return None
                        def __exit__(self_inner, exc_type, exc_val, exc_tb):
                            return False
                    return lambda *a, **k: Dummy()
                return lambda *a, **k: None

        st = DummyModule('streamlit')
        st.sidebar = DummyModule('streamlit.sidebar')
        st.session_state = {}
        st.__version__ = '0'
        ensure_module('streamlit', st)

    try:
        import numpy  # type: ignore
    except Exception:
        class NumpyStub(types.ModuleType):
            def __getattr__(self, name):
                if name == 'sqrt':
                    return math.sqrt
                if name == 'arcsin':
                    return math.asin
                if name == 'ndarray':
                    class NDArray:  # minimal placeholder type
                        pass
                    return NDArray
                if name == 'isscalar':
                    return lambda obj: False
                return lambda *a, **k: None

        np = NumpyStub('numpy')
        np.__version__ = '0'
        ensure_module('numpy', np)

    try:
        import pandas  # type: ignore
    except Exception:
        class PandasStub(types.ModuleType):
            def __getattr__(self, name):
                return lambda *a, **k: None

        pd = PandasStub('pandas')
        pd.__version__ = '0'
        ensure_module('pandas', pd)

    try:
        from scipy.stats import norm  # noqa: F401
    except Exception:
        scipy_mod = types.ModuleType('scipy')
        scipy_mod.__version__ = '0'
        stats_mod = types.ModuleType('scipy.stats')
        stats_mod.norm = types.SimpleNamespace(ppf=lambda *a, **k: 0.0,
                                               cdf=lambda *a, **k: 0.0)
        scipy_mod.stats = stats_mod
        ensure_module('scipy', scipy_mod)
        ensure_module('scipy.stats', stats_mod)

    try:
        from statsmodels.stats.power import TTestIndPower  # noqa: F401
    except Exception:
        sm_mod = types.ModuleType('statsmodels')
        sm_mod.__version__ = '0'
        sm_stats = types.ModuleType('statsmodels.stats')
        sm_power = types.ModuleType('statsmodels.stats.power')
        class Dummy:  # minimal stand-in classes
            pass
        sm_power.TTestIndPower = Dummy
        sm_power.TTestPower = Dummy
        sm_power.FTestAnovaPower = Dummy
        sm_stats.power = sm_power
        sm_mod.stats = sm_stats
        ensure_module('statsmodels', sm_mod)
        ensure_module('statsmodels.stats', sm_stats)
        ensure_module('statsmodels.stats.power', sm_power)

    # Load the target module using the created stubs so that ``import`` calls
    # inside ``power_sample_calc.py`` resolve to the dummy modules above.
    spec = importlib.util.spec_from_file_location(
        'power_sample_calc',
        str((Path(__file__).resolve().parents[1] / 'power_sample_calc.py')),
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules['power_sample_calc'] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def calc():
    mod = import_power_sample_calc()
    return mod.calculate_effect_size


def test_cohen_d_two(calc):
    result = calc('cohen_d_two', mean1=5, mean2=10, pooled_sd=2)
    assert result == pytest.approx(2.5)

    result = calc('cohen_d_two', mean1=8, mean2=5, pooled_sd=3)
    assert result == pytest.approx(1.0)

    assert calc('cohen_d_two', mean1=5, mean2=5, pooled_sd=0) is None


def test_cohen_d_one(calc):
    result = calc('cohen_d_one', sample_mean=5, hypothesized_mean=7, sd=2)
    assert result == pytest.approx(1.0)

    assert calc('cohen_d_one', sample_mean=5, hypothesized_mean=5, sd=0) is None


def test_cohen_d_paired(calc):
    result = calc('cohen_d_paired', mean_diff=-4, sd_diff=2)
    assert result == pytest.approx(2.0)

    assert calc('cohen_d_paired', mean_diff=4, sd_diff=0) is None


def test_cohen_h(calc):
    result = calc('cohen_h', p1=0.6, p2=0.4)
    expected = abs(2 * math.asin(math.sqrt(0.6)) - 2 * math.asin(math.sqrt(0.4)))
    assert result == pytest.approx(expected)

    assert calc('cohen_h', p1=0.4, p2=0.4) is None
    assert calc('cohen_h', p1=0.0, p2=0.4) is None
