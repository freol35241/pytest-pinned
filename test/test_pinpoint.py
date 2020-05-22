import pytest
import json

def test_passing_with_pinned_results(testdir):
    """Testing that we fail tests prior to having any 
    expected results to compare with.
    """

    # create a temporary pytest test file
    testdir.makepyfile(
        """
        def test_str(pinned):
            assert pinned == "Hello World!"
            
        def test_scalar(pinned):
            assert 3.0 == pinned
            
        def test_list(pinned):
            assert [[1,2,3]] == pinned
            
        def test_dict(pinned):
            assert {'a': 1, 'b': 2, 'c': 3} == pinned
            
        def test_multiple(pinned):
            assert pinned == "Hello World!"
            assert 3.0 == pinned
            assert [[1,2,3]] == pinned
            assert {'a': 1, 'b': 2, 'c': 3} == pinned

    """
    )
    
    # Collect expected results
    result = testdir.runpytest('--pinned-rewrite')
    result.assert_outcomes(passed=5)
    
    # Test again, this time ot should pass
    result = testdir.runpytest()
    result.assert_outcomes(passed=5)
    
def test_failing_with_pinned_results(testdir):
    """Testing that we fail tests prior to having any 
    expected results to compare with.
    """

    # create a temporary pytest test file
    testdir.makepyfile(
        """
        from random import random
            
        def test_scalar(pinned):
            assert random() == pinned
            
        def test_list(pinned):
            assert [[random(), random(), random()]] == pinned
            
        def test_dict(pinned):
            assert {'a': random(), 'b': random(), 'c': random()} == pinned

    """
    )

    # Collect expected results
    result = testdir.runpytest('--pinned-rewrite')
    result.assert_outcomes(passed=3)
    
    # Test again, using the pinned results, should still fail!
    result = testdir.runpytest()
    result.assert_outcomes(failed=3)
    
def test_approx_passing_with_pinned_results(testdir):
    """Testing that we fail tests prior to having any 
    expected results to compare with.
    """

    # create a temporary pytest test file
    testdir.makepyfile(
        """
        from random import random
            
        def test_scalar(pinned):
            assert random() == pinned(abs=1)
            
        def test_list(pinned):
            assert [random(), random(), random()] == pinned(abs=1)
            
        def test_dict(pinned):
            assert {'a': random(), 'b': random(), 'c': random()} == pinned(abs=1)

    """
    )

    # Collect expected results
    result = testdir.runpytest('--pinned-rewrite')
    result.assert_outcomes(passed=3)
    
    # Test again, using the pinned results, should still fail!
    result = testdir.runpytest()
    result.assert_outcomes(passed=3)
