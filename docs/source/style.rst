.. _style-source:

Style and conventions
=====================

- All new element (class, member of a class, struct, function) declared in a .H file must have a Doxygen-readable documentation
- Indent four spaces
- No tabs allowed
- No end-of-line whitespaces allowed
- Classes use CamelCase
- Objects use snake_case
- Lines should not have >100 characters
- The declaration and definition of a function should have a space between the function name and the first bracket (``my_function (...)``), function calls should not (``my_function(...)``).
  This is a convention introduce in AMReX so ``git grep "my_function ("`` returns only the declaration and definition, not the many function calls.
