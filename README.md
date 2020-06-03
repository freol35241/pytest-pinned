pytest-pinned
===============
[![PyPI version shields.io](https://img.shields.io/pypi/v/pytest-pinned.svg)](https://pypi.python.org/pypi/pytest-pinned/)
![pinned](https://github.com/freol35241/pytest-pinned/workflows/pinned/badge.svg)

A simple [`pytest`](https://docs.pytest.org/en/latest/) plugin for writing **pinning tests**.

Pinning tests or snapshot tests or [characterization tests](https://en.wikipedia.org/wiki/Characterization_test) are meant to describe (characterize) the actual behavior of an existing piece of software, and therefore protect existing behavior against unintended changes via automated testing.

This type of testing can come in handy for several reasons:
* Legacy codebase with poor or non-existing test coverage.
* As broad integration tests where more specific testing may prove difficult.
* Scientific computing where implementation of, usually quite complex, scientific models are hard to test for specific behaviours, especially when they are used for research purposes. 
* As extra confidence boosters during refactoring. 

`pytest-pinned` keeps all expected results from pinning tests in a single, pretty-printed `JSON` file resulting in only a single file needing to be added to the [VCS](https://en.wikipedia.org/wiki/Version_control) repository and diffs are also contained to this single file. The use of `JSON` for serialization of the expected results however imposes some restrictions on the datatypes that can be used, see the [JSON type conversion table](https://docs.python.org/3/library/json.html#py-to-json-table) from the standard library. In addition, however, `pytest-pinned>=0.2.0` natively supports `numpy` arrays!

Note: `pytest-pinned` is not compatible with [pytest-xdist](https://pypi.org/project/pytest-xdist/) or any other plugin that runs test in separate subprocesses.


### Requirements

`pytest-pinned` has no external dependencies except for [`pytest`](https://docs.pytest.org/en/latest/) itself.


### Installation

You can install `pytest-pinned` via `pip` from `PyPI`:

    $ pip install pytest-pinned


### Usage

`pytest-pinned` expose a single pytest fixture (`pinned`) with a very simple syntax. `pinned` will keep track of what test it is used in, supports usage with the standard `assert` statement and allows for multiple asserts in the same test.

#### Syntax

Simple pinning test sample:
```
def test_simple(pinned):
    assert(10.0 == pinned)
```

`pytest-pinned` also supports approximate comparisons using [`pytest.approx`](https://docs.pytest.org/en/latest/reference.html#pytest-approx). See last assert statement in example below for syntax. `pinned` accepts the same keyword arguments as `pytest.approx`.

More elaborate example:
```
def test_elaborate(pinned):
    assert(10.0 == pinned)
    assert([1,2,3] == pinned)
    assert({'a': 1, 'b': 2} == pinned)
    assert(5.2983746239134 == pinned.approx(rel=0.00001, abs=0.001))
```
#### Expected results

If `pytest-pinned` cannot find any expected results for a comparison it will fail the test and ask teh user to write new expected results.

To rewrite the expected results "from scratch", use:

    $ pytest --pinned-rewrite

To update the expected results for only some tests, use:

    $ pytest tests/sample_test.py::specific_test --pinned-update

To change the path where `pytest-pinned` stores (and loads) the expected results, use:

    $ pytest --pinned-path path/to/expected/results.json

### License

Distributed under the terms of the `MIT` license, `pytest-pinned` is free and open source software

### Issues

If you encounter any problems, please [`file an issue`](https://github.com/freol35241/pytest-pinned/issues) along with a detailed description.

### Contributing

Contributions are very welcome.
