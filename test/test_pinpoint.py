import pytest
import json
import numpy as np

def test_failing_without_pinned_results(testdir):
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
    
    # Run tests without any collected results, should fail
    result = testdir.runpytest()
    result.assert_outcomes(failed=5)

def test_passing_with_pinned_results(testdir):
    """Testing that we pass when we have collected results
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
    """Testing that we fail tests when results are not what we expect
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
    """Testing that we pass test when results are approximately what we expect
    """

    # create a temporary pytest test file
    testdir.makepyfile(
        """
        from random import random
            
        def test_scalar(pinned):
            assert random() == pinned.approx(abs=1)
            
        def test_list(pinned):
            assert [random(), random(), random()] == pinned.approx(abs=1)
            
        def test_dict(pinned):
            assert {'a': random(), 'b': random(), 'c': random()} == pinned.approx(abs=1)

    """
    )

    # Collect expected results
    result = testdir.runpytest('--pinned-rewrite')
    result.assert_outcomes(passed=3)
    
    # Test again, using the pinned results, should still fail!
    result = testdir.runpytest()
    result.assert_outcomes(passed=3)
    
def test_numpy_combos(testdir):
    """Testing that we maintain compatibility with numpy
    """

    # create a temporary pytest test file
    testdir.makepyfile(
        """
        def test_array(pinned):
            import sys
            np = sys.modules.get('numpy')
            assert np.ones((5,7)) == pinned
            
        def test_multiple_asserts(pinned):
            import sys
            np = sys.modules.get('numpy')
            assert (np.ones((4, 4)) == np.ones((4, 4))).all()
            assert np.ones((5,7)) == pinned.approx()

    """
    )

    # Collect expected results
    result = testdir.runpytest('--pinned-rewrite')
    result.assert_outcomes(passed=2)
    
    # Test again, using the pinned results, should still fail!
    result = testdir.runpytest()
    result.assert_outcomes(passed=2)

def test_stdout_when_pinned_assert_fails(testdir):
    """Related to issue #7, we need to inform the user that it wasnt our fault!
    """
    
    # create a temporary pytest test file
    testdir.makepyfile(
        """
        
        def test_str(pinned):
            assert pinned == "Hello World!"

    """
    )
    
    # Collect expected results
    result = testdir.runpytest("--pinned-rewrite")
    result.assert_outcomes(passed=1)
    
    # rewrite the test file with some funky results
    testdir.makepyfile(
        """
        
        def test_str(pinned):
            assert pinned == "Hola World!"

    """
    )
    
    # Run tests again
    result = testdir.runpytest()
    result.assert_outcomes(failed=1)
    
    assert "The pinned fixture was not used yet in this test." not in result.stdout.str()
    
def test_stdout_when_other_assert_fails_first(testdir):
    """Related to issue #7, we need to inform the user that it wasnt our fault!
    """
    
    # create a temporary pytest test file
    testdir.makepyfile(
        """
        
        def test_str(pinned):
            assert pinned == "Hello World!"

    """
    )
    
    # Collect expected results
    result = testdir.runpytest("--pinned-rewrite")
    result.assert_outcomes(passed=1)
    
    # Insert AssertionError prior to pinned usage
    testdir.makepyfile(
        """
        
        def test_str(pinned):
            assert False
            assert pinned == "Hello World!"

    """
    )
    
    # Collect expected results
    result = testdir.runpytest()
    result.assert_outcomes(failed=1)
    assert "The pinned fixture was not used yet in this test." in result.stdout.str()
    
def test_stdout_when_other_assert_fails_second(testdir):
    """Related to issue #7, we need to inform the user that it wasnt our fault!
    """
    
    # create a temporary pytest test file
    testdir.makepyfile(
        """
        
        def test_str(pinned):
            assert pinned == "Hello World!"

    """
    )
    
    # Collect expected results
    result = testdir.runpytest("--pinned-rewrite")
    result.assert_outcomes(passed=1)
    
    # Insert AssertionError after pinned usage
    testdir.makepyfile(
        """
        
        def test_str(pinned):
            assert pinned == "Hello World!"
            assert False

    """
    )
    
    # Collect expected results
    result = testdir.runpytest()
    result.assert_outcomes(failed=1)
    assert "pinned = Pinned(Hello World!)" in result.stdout.str()