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

Current empty modules:

* src.flows.eval_flows

Known Issues
------------
Some modules are not compiled by autodoc, therefor may not documented. The issue comes from the fact
that while compiling, the environment that the doc is compiled in, has to have the required packages used
in the documented code. Not having them installed results in an empty doc page.

If one wants to rebuild the html via ``./docs./make.bat html`` (on Windows) or ``make html`` (on Linux) make sure
to have the required packages used in the source code!

``./docs/auto_make_html.sh`` also builds the doc with added logging functionality!

Indices and tables
------------------
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`