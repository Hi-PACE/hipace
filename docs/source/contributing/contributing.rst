Contribute to Hipace++
======================

We welcome new contributors!

To contribute to Hipace++, the steps are as follows:
 - Fork the Hipace++ repo, so you have your own fork
 - Pull the latest development from baseline, and create your ``<new branch>`` from it
 - Commit your changes as usual, and push them on your fork
 - Open a PR between ``<new branch>`` on your for and ``development`` on baseline

Documentation
-------------

Hipace++ has a full (functions and classes and their members, albeit sometimes basic) Doxygen-readable documentation. You can compile it with

.. code-block:: bash

   cd docs
   doxygen
   open doxyhtml/index.html

The last line would work on MacOS. On another platform, open the html file with your favorite browser.

The Hipace++ Doxygen documentation can be found `here <../_static/doxyhtml/index.html>`__.
   
Style and conventions
---------------------

- All new element (class, member of a class, struct, function) declared in a .H file must have a Doxygen-readable documentation
- Indent four spaces
- No tabs allowed
- No end-of-line whitespaces allowed
- Classes use CamelCase
- Objects use snake_case
- Lines should not have >100 characters
- The declaration and definition of a function should have a space between the function name and the first bracket (``my_function (...)``), function calls should not (``my_function(...)``).
  This is a convention introduce in AMReX so ``git grep "my_function ("`` returns only the declaration and definition, not the many function calls.
