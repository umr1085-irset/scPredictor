def printd(d, indent=0):
    '''
    Print dictionaries
    
    Parameters
    ----------
    d : dict
        A dictionary
    indent : int
        Number of indents to add
    '''
    for key, value in d.items():
        print('\t' * indent + str(key) + f': {value}')
        