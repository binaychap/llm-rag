import pytest

from basic_python_project import hello

def test_hello():
    assert hello() == "Hello from basic Python project!"