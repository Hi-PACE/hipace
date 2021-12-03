Contribute to HiPACE++
======================

We welcome new contributors!

To contribute to HiPACE++, the steps are as follows:
 - Fork the HiPACE++ repo, so you have your own fork
 - Pull the latest development from baseline, and create your ``<new branch>`` from it
 - Commit your changes as usual, and push them on your fork
 - Open a PR between ``<new branch>`` on your for and ``development`` on baseline

Documentation
-------------

HiPACE++ has a full (functions and classes and their members, albeit sometimes basic) Doxygen-readable documentation. You can compile it with

.. code-block:: bash

   cd docs
   doxygen
   open doxyhtml/index.html

The last line would work on MacOS. On another platform, open the html file with your favorite browser.

The HiPACE++ Doxygen documentation can be found `here <../_static/doxyhtml/index.html>`__.

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

How-to
------

Make a new release
~~~~~~~~~~~~~~~~~~

- Find the release tag in all files with something like ``git grep '21.12'`` and modify where relevant (be careful with automated search & replace operations, they may cause unwanted changes).
- On the main repo page, go to Releases > Draft new release, and
    * Update the AMReX and openPMD-api versions
    * Click button "Auto-generate release notes" to get a well-formatted list of PRs
    * Update the commands that you used
    * Add any additional comments
    * confirm the release
- Once the release is done, Zenodo will generate a DOI. Go to zenodo.org > My Account > Github > HiPACE++, and get the DOI of the last release and copy-paste to the release description
