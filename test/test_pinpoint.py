import pytest
import json

def test_passing_with_pinned_results(testdir):
    """Testing that we fail tests prior to having any 
    expected results to compare with.
    """

    # create a temporary pytest test file
    testdir.makepyfile(
        """
        def test_str(pinpointed):
            assert pinpointed == "Hello World!"
            
        def test_scalar(pinpointed):
            assert 3.0 == pinpointed
            
        def test_list(pinpointed):
            assert [[1,2,3]] == pinpointed
            
        def test_dict(pinpointed):
            assert {'a': 1, 'b': 2, 'c': 3} == pinpointed
            
        def test_multiple(pinpointed):
            assert pinpointed == "Hello World!"
            assert 3.0 == pinpointed
            assert [[1,2,3]] == pinpointed
            assert {'a': 1, 'b': 2, 'c': 3} == pinpointed

    """
    )
    
    # Collect expected results
    result = testdir.runpytest('--pinpoint-rewrite')
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
            
        def test_scalar(pinpointed):
            assert random() == pinpointed
            
        def test_list(pinpointed):
            assert [[random(), random(), random()]] == pinpointed
            
        def test_dict(pinpointed):
            assert {'a': random(), 'b': random(), 'c': random()} == pinpointed

    """
    )

    # Collect expected results
    result = testdir.runpytest('--pinpoint-rewrite')
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
            
        def test_scalar(pinpointed):
            assert random() == pinpointed(abs=1)
            
        def test_list(pinpointed):
            assert [random(), random(), random()] == pinpointed(abs=1)
            
        def test_dict(pinpointed):
            assert {'a': random(), 'b': random(), 'c': random()} == pinpointed(abs=1)

    """
    )

    # Collect expected results
    result = testdir.runpytest('--pinpoint-rewrite')
    result.assert_outcomes(passed=3)
    
    # Test again, using the pinned results, should still fail!
    result = testdir.runpytest()
    result.assert_outcomes(passed=3)
