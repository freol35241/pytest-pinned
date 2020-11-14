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
            
def _is_numpy_array(obj):
    import sys
    np = sys.modules.get("numpy")
    if np:
        return isinstance(obj, np.ndarray)
    return False

class ExpectedResult:
    
    # Tell numpy to use our `__eq__` operator instead of its.
    
    __array_ufunc__ = None
    __array_priority__ = 100

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
        if self._idx:
            return self._assemble_key(self._idx)
        return False

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
        
        # Special treatment of numpy arrays
        import sys
        np = sys.modules.get("numpy")
        array = np and isinstance(other, np.ndarray)
        
        if self._write:
            self._expected[key] = other if not array else other.tolist()
        
        expected = self._get_expected(key)
        res = self._compare_func(expected, other)
        
        res = np.all(res) if array else res
        
        self._reset_compare_func()
        
        return res
    
    def approx(self, *args, **kwargs):        
        def wrapper(expected, value):
            return eq(expected, pytest.approx(value, *args, **kwargs))
        self._compare_func = wrapper
        return self
    
    def __repr__(self):
        key = self._get_current_key()
        if not key:
            # This needs to be handled since pytest apparently calls __repr__ on this
            # object even if an earlier assert statement failed
            return "The pinned fixture was not used yet in this test."
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
