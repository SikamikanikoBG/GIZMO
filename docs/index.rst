.. Gizmo documentation master file, created by
   sphinx-quickstart on Tue Jun 11 10:10:31 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Gizmo's documentation!
=================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules

Current non-empty modules:

* api
* api.module
* some of src

Known Issues
------------
Some modules are not compiled by autodoc, therefor may not documented. The issue comes from the fact
that while compiling, the environment that the doc is compiled in, has to have the required packages used
in the documented code. Not having them installed results in an empty doc page.

If one wants to rebuild the html via ``./docs./make.bat html`` make sure
to have the required packages used in the source code!

Other errors include missing files and bad indexing. Examples:

* ``FileNotFoundError: [Errno 2] No such file or directory: './input_data/input_source_original.parquet'``.
* ``FileNotFoundError: [Errno 2] No such file or directory: 'gizmo_logo.png'``.
* ``TypeError: 'type' object is not subscriptable``.

Full error message in ``./docs/logs/log_11_06_2024_1303_manual.txt``

Indices and tables
------------------
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
