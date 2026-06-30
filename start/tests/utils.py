import importlib.util
import pathlib
import sys
import types
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load_module_from_path(module_name: str, path: pathlib.Path):
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    spec = importlib.util.spec_from_file_location(module_name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_student_client_module(module_name: str = "student_client_under_test"):
    return _load_module_from_path(module_name, ROOT / "student_fl_client_grpc.py")


def load_student_server_module(module_name: str = "student_server_under_test"):
    return _load_module_from_path(module_name, ROOT / "student_fl_server_grpc.py")


class DummyContextManager:
    """Pequeno helper para simular `with grpc.insecure_channel(...) as channel:` nos testes."""

    def __init__(self, value: Any):
        self.value = value

    def __enter__(self):
        return self.value

    def __exit__(self, exc_type, exc, tb):
        return False
