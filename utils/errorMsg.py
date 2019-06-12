def assertLengthError(target, length, method='exactly'):
    '''
    Assert an error of length over range of a list or a tuple

    method: 'exactly' or 'range'
    '''
    methods = ['exactly', 'range']
    methods.index(method)

    if(method == 'exactly'):
        assert len(target) == length, 'Required a list or a tuple of length {}, but receive a {}'.format(length, target)
    else:
        assert len(length) != 2, 'Required a list or a tuple of length 2, but receive a {}'.format(length)
        assert length[0] <= len(target) <= length[1], 'Required a list or a tuple of length {}, but receive a {}'.format(length, target)

def assertTypeError(target, targetType):
    '''
    TODO
    '''
    assert type(target) == targetType, 'Required a type of {}, but receive a {}'.format(targetType, type(target))

def assertRangeError(target, ranges):
    '''
    TODO
    '''
    assert ranges[0] <= target <= ranges[1], 'Required a number in range {}, but receive a {}'.format(ranges, target)
