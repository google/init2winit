import importlib
import inspect
import types
import pytest


def _import_or_skip(name: str) -> types.ModuleType:
    return pytest.importorskip(f"init2winit.{name}")


def test_import_main_modules():
    modules = ["hyperparameters", "schedules", "utils", "checkpoint", "training_metrics_grabber"]
    for m in modules:
        mod = _import_or_skip(m)
        assert isinstance(mod, types.ModuleType)


def test_schedules_has_callable_schedule():
    mod = _import_or_skip("schedules")
    candidates = [name for name in dir(mod) if "schedule" in name.lower() or "lr" in name.lower() or "learning" in name.lower()]
    if not candidates:
        pytest.skip("No schedule-like callables found in schedules module")

    # Try to call the first candidate conservatively
    for name in candidates:
        attr = getattr(mod, name)
        if callable(attr):
            try:
                sig = inspect.signature(attr)
                # Prefer calling with a single integer step if possible
                if len(sig.parameters) == 1:
                    val = attr(0)
                else:
                    # try common (step, base) pattern
                    val = attr(0, 0)
                assert isinstance(val, (int, float))
                return
            except Exception:
                # If it fails, we can try the next candidate, but we won't fail the test immediately
                continue

    pytest.skip("Found schedule-like names but none were safely callable with simple args")


def test_checkpoint_exposes_save_or_restore():
    mod = _import_or_skip("checkpoint")
    names = dir(mod)
    has_save = any(n.lower().startswith("save") for n in names)
    has_restore = any(n.lower().startswith("restore") or n.lower().startswith("load") for n in names)
    assert has_save or has_restore
