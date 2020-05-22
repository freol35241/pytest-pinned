pytest-pinpoint
===============

A simple [`pytest`](https://docs.pytest.org/en/latest/) plugin for writing pinning tests.

Pinning tests or snapshot tests or [characterization tests](https://en.wikipedia.org/wiki/Characterization_test) are meant to describe (characterize) the actual behavior of an existing piece of software, and therefore protect existing behavior against unintended changes via automated testing.

This type of testing can come in handy for several reasons:
* Legacy codebase with poor or non-existing test coverage
* As broad integration tests where more specific testing may prove difficult

In my case, the main usecase is legacy and non-legacy codebases aimed for scientific computing. The implementation of, usually quite complex, scientific models are hard to test for specific behaviours, especially when they are used for research purposes. Pinning tests allow for more confident refactoring of the implementation of such models while tracing how results and predicitons changes.

`pytest-pinpoint` keeps all expected results from pinning tests in a single `JSON` file resulting in only a single file needing to be added to the [VCS](https://en.wikipedia.org/wiki/Version_control) repository and diffs are also contained to this single file. The use of `JSON` for serialization of the expected results however imposes some restrictions on the datatypes that can be used, see the [JSON type conversion table](https://docs.python.org/3/library/json.html#py-to-json-table) from the standard library.


### Requirements

`pytest-pinpoint` has no external dependencies except for [`pytest`](https://docs.pytest.org/en/latest/) itself.


### Installation

You can install `pytest-pinpoint` via `pip` from `PyPI`:

    $ pip install pytest-pinpoint


Usage
-----

* TODO

Contributing
------------
Contributions are very welcome.

License
-------

Distributed under the terms of the `MIT` license, `pytest-pinpoint` is free and open source software


Issues
------

If you encounter any problems, please `file an issue` along with a detailed description.
