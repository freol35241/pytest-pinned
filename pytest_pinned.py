import json
import difflib
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
            
def is_ndarray(obj):
    import sys
    np = sys.modules.get("numpy")
    
    return np and isinstance(obj, np.ndarray)

def un_numpify(obj):
    if is_ndarray(obj):
        return obj.tolist()
    
    return obj

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
        
        if self._write:
            self._expected[key] = un_numpify(other)
        
        expected = self._get_expected(key)
        res = self._compare_func(expected, other)
        
        if is_ndarray(res):
            res = res.all()
        
        self._reset_compare_func()
        
        return res
    
    def approx(self, *args, **kwargs):        
        def wrapper(expected, value):
            return eq(expected, pytest.approx(value, *args, **kwargs))
        self._compare_func = wrapper
        return self
    
    def __repr__(self):
        return "Pinned({})".format(self._node.nodeid)


@pytest.fixture
def pinned(request):
    write = request.config.getoption('rewrite') \
        or request.config.getoption('update')

    return ExpectedResult(
        EXPECTED_RESULTS,
        request.node,
        write)
    
def pytest_assertrepr_compare(config, op, left, right):
    if op != "==":
        return
    
    if isinstance(left, ExpectedResult):
        expected = left._get_expected(left._get_current_key())
        actual = un_numpify(right)
    elif isinstance(right, ExpectedResult):
        expected = right._get_expected(right._get_current_key())
        actual = un_numpify(left)
    else:
        return None
    
    expected = json.dumps(expected, indent=4, sort_keys=True).splitlines()
    actual = json.dumps(actual, indent=4, sort_keys=True).splitlines()
    
    diff = list(difflib.unified_diff(expected, actual, fromfile="Expected", tofile="Actual", lineterm=""))
    diff.insert(0, "")
    return diff
