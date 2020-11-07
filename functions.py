from itertools import zip_longest

def bytes_group(n, iterable, fillvalue=None):
    return zip_longest(*[iter(iterable)]*n, fillvalue=fillvalue)
