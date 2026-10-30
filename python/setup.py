"""The wheel carries coralrt's native library, which Python loads through ctypes: a platform wheel that any Python 3
can use (py3-none-<platform>). The metadata is in pyproject.toml; build_wheel.py makes the wheel."""
from setuptools import setup
from setuptools.command.bdist_wheel import bdist_wheel
from setuptools.dist import Distribution


class _Binary(Distribution):
  def has_ext_modules(self):
    return True


class _AnyPython(bdist_wheel):
  def get_tag(self):
    return ("py3", "none", super().get_tag()[2])


setup(distclass=_Binary, cmdclass={"bdist_wheel": _AnyPython})
