#! /usr/bin/env python
"""
Convert empty IPython notebook to a sphinx doc page.

Derived from Michael Waskom's Seaborn package:

https://github.com/mwaskom/seaborn/blob/master/doc/tutorial/tools/nb_to_doc.py
"""
import sys
import os
from subprocess import check_call as sh


def convert_nb(nbname):

    if not nbname.endswith('.ipynb'):
        nbname = nbname + '.ipynb'
    rst = nbname.replace('.ipynb', '.rst')

    # Return if .rst file already exists and is newer than the notebook:
    if (os.path.exists(rst) and
            os.path.getmtime(rst) > os.path.getmtime(nbname)):
        return

    d, b = os.path.split(nbname)
    tmp = os.path.join(d, 'temp_'+b)
    tmp2 = 'temp_' + rst

    # Execute the notebook:
    sh(["jupyter", "nbconvert", "--to", "notebook",
        "--execute", nbname, "--output", tmp])

    # Convert to .rst for Sphinx:
    sh(["jupyter", "nbconvert", "--to", "rst", tmp, "--output", tmp2])

    with open(tmp2) as fin:
        with open(rst, 'wt') as fout:
            for line in fin:
                fout.write(line.replace('.. parsed-literal::',
#                                        '::'))
#                                        '.. code::'))
                                        '.. code-block:: none'))
#                                        '.. literal::'))

    # Remove the temporary notebooks:
    os.remove(tmp)
    os.remove(tmp2)


if __name__ == "__main__":

    for nbname in sys.argv[1:]:
        convert_nb(nbname)
