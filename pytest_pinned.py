import json
from pathlib import Path
from operator import eq

import pytest

EXPECTED_RESULTS = dict()

def pytest_addoption(parser):
    group = parser.getgroup("pinned")
    
    group.addoption(
        '--pinned-path',
        dest='path',
        type=Path,
        default=Path('pinned_results.json'),
        help='Path to file where expected results will be stored' 
    )
    
    group.addoption(
        '--pinned-update',
        dest='update',
        action='store_true',
        default=False,
        help='Updates expected results.',
    )
    
    group.addoption(
        '--pinned-rewrite',
        dest='rewrite',
        action='store_true',
        default=False,
        help='Writes new results to blank file.',
    )

def pytest_configure(config):
    global EXPECTED_RESULTS
    path = config.getoption('path')
    rewrite = config.getoption('rewrite')

    if path.exists() and not rewrite:
        with path.open('r') as f:
            EXPECTED_RESULTS = json.load(f)

def pytest_unconfigure(config):
    path = config.getoption('path')

    if EXPECTED_RESULTS:
        with path.open('w') as f:
            json.dump(EXPECTED_RESULTS, f, indent=4, sort_keys=True)

class ExpectedResult:

    def __init__(self, expected, node, write):
        self._expected = expected
        self._node = node
        self._write = write
        self._idx = 0
        self._reset_compare_func()
        
    def _reset_compare_func(self):
        self._compare_func = eq
        
    def _assemble_key(self, id):
        key = '{}_{}'.format(
            self._node.nodeid,
            id
        )
        return key
        
    def _get_current_key(self):
        return self._assemble_key(self._idx)

    def _get_next_key(self):
        self._idx += 1
        return self._get_current_key()
    
    def _get_expected(self, key):
        try:
            return self._expected[key]
        except KeyError:
            pytest.fail(
                'Node with nodeid: {} does not have a stored value to compare with!'
                ' Please use --pinned-update or --pinned-rewrite first.'.format(key),
                pytrace=False
                )

    def __eq__(self, other):
        key = self._get_next_key()
        
        if self._write:
            self._expected[key] = other
            return True
        
        expected = self._get_expected(key)
        res = self._compare_func(expected, other)
        
        self._reset_compare_func()
        
        return res
    
    def __call__(self, *args, **kwargs):
        def approx_wrapper(expected, value):
            return eq(expected, pytest.approx(value, *args, **kwargs))
        self._compare_func = approx_wrapper
        return self
    
    def __repr__(self):
        key = self._get_current_key()
        expected = self._get_expected(key)
        return 'Pinned({})'.format(expected)

@pytest.fixture
def pinned(request):
    write = request.config.getoption('rewrite') \
        or request.config.getoption('update')

    return ExpectedResult(
        EXPECTED_RESULTS,
        request.node,
        write)
