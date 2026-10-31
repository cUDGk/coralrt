"""The wheel carries coralrt's native library, which Python loads through ctypes: a platform wheel that any Python 3
can use (py3-none-<platform>). The metadata is in pyproject.toml; build_wheel.py makes the wheel."""
import pathlib
import re

from setuptools import setup
from setuptools.command.bdist_wheel import bdist_wheel
from setuptools.dist import Distribution


class _Binary(Distribution):
  def has_ext_modules(self):
    return True


class _AnyPython(bdist_wheel):
  def get_tag(self):
    return ("py3", "none", super().get_tag()[2])


# PyPI shows the English README, whose links are relative to the repository
readme = (pathlib.Path(__file__).resolve().parent.parent / "README_en.md").read_text(encoding="utf-8")
readme = re.sub(r"\]\((?!https?:|#)", "](https://github.com/cUDGk/coralrt/blob/main/", readme)
setup(distclass=_Binary, cmdclass={"bdist_wheel": _AnyPython}, long_description=readme,
      long_description_content_type="text/markdown")
