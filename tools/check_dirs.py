'''helper script to check which directories are not on python path

all test folders should have and ``__init__.py``

'''

import os

root = '../statsmodels'

for base, dnames, fnames in os.walk(root):
    if not '__init__.py' in fnames:
        #I have some empty directories when I switch git branches
        if (len(dnames) + len(fnames)) != 0:
            print(base, dnames, len(fnames))

    if '__pycache__' in dnames:
        dnames.remove('__pycache__')
    if 'src' in dnames:
        dnames.remove('src')
