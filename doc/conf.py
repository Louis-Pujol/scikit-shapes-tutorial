from __future__ import annotations
import importlib.metadata

# -- pyvista configuration ---------------------------------------------------
# See: https://github.com/pyvista/pyvista/blob/main/doc/source/conf.py
import pyvista
import os
from pyvista.core.errors import PyVistaDeprecationWarning
from pyvista.core.utilities.docs import linkcode_resolve, pv_html_page_context  # noqa: F401
from pyvista.plotting.utilities.sphinx_gallery import DynamicScraper as Scraper

# Manage errors
pyvista.set_error_output_file("errors.txt")
# Ensure that offscreen rendering is used for docs generation
pyvista.OFF_SCREEN = True  # Not necessary - simply an insurance policy
# Preferred plotting style for documentation
pyvista.set_plot_theme("document")
pyvista.global_theme.window_size = [1024, 768]
pyvista.global_theme.font.size = 22
pyvista.global_theme.font.label_size = 22
pyvista.global_theme.font.title_size = 22
pyvista.global_theme.return_cpos = False
pyvista.set_jupyter_backend(None)
# Save figures in specified directory
pyvista.FIGURE_PATH = os.path.join(os.path.abspath("./images/"), "auto-generated/")
if not os.path.exists(pyvista.FIGURE_PATH):
    os.makedirs(pyvista.FIGURE_PATH)

# necessary when building the sphinx gallery
pyvista.BUILDING_GALLERY = True
os.environ['PYVISTA_BUILDING_GALLERY'] = 'true'

class ResetPyVista:
    """Reset pyvista module to default settings."""

    def __call__(self, gallery_conf, fname):
        """Reset pyvista module to default settings

        If default documentation settings are modified in any example, reset here.
        """
        import pyvista

        pyvista._wrappers['vtkPolyData'] = pyvista.PolyData
        pyvista.set_plot_theme('document')

    def __repr__(self):
        return 'ResetPyVista'


reset_pyvista = ResetPyVista()


project = "Scikit Shapes tutorial"
copyright = "2024, Louis Pujol"
author = "Louis Pujol"
version = release = ""

extensions = [
    # 'sphinx_tabs.tabs',
    "sphinx_design",
    # "myst_parser",
    "pyvista.ext.plot_directive",
    "pyvista.ext.viewer_directive",
    # "sphinx.ext.autodoc",
    # "sphinx.ext.autosummary",
    # "numpydoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    # "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    'sphinx_gallery.gen_gallery',
]

source_suffix = [".rst"]
exclude_patterns = [
    "_build",
    "**.ipynb_checkpoints",
    "Thumbs.db",
    ".DS_Store",
    ".env",
    ".venv",
]

sphinx_tabs_valid_builders = ['linkcheck']

html_theme = "furo"

myst_enable_extensions = [
    "colon_fence",
]

sphinx_gallery_conf = {
    'examples_dirs': '../examples/',   # path to your example scripts
    'gallery_dirs': 'auto_examples',  # path to where to save gallery generated output
    "doc_module": "pyvista",
    "image_scrapers": (Scraper(), "matplotlib"),
    "first_notebook_cell": "%matplotlib inline",
    "reset_modules": (reset_pyvista,),
    "reset_modules_order": "both",
}
