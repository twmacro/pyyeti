#! /usr/bin/env python
"""
Convert empty IPython notebook to a sphinx doc page.

Derived from Michael Waskom's Seaborn package:

https://github.com/mwaskom/seaborn/blob/master/doc/tutorial/tools/nb_to_doc.py
"""
import sys
import os
from subprocess import run


def convert_nb(nbname):
    if not nbname.endswith(".ipynb"):
        nbname = nbname + ".ipynb"
    rst = nbname.replace(".ipynb", ".rst")

    # Return if .rst file already exists and is newer than the notebook:
    if os.path.exists(rst) and os.path.getmtime(rst) > os.path.getmtime(nbname):
        return

    d, b = os.path.split(nbname)
    tmp = os.path.join(d, "temp_" + b)
    d, b = os.path.split(rst)
    tmp2 = os.path.join(d, "temp_" + b)

    # Execute the notebook:
    cmd = [
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        nbname,
        "--output",
        tmp,
    ]

    print(f"running: {cmd}")
    run(cmd, check=True)

    # I don't understand, but on Read the Docs, output name was
    # 'temp_ode.nbconvert.ipynb' (for example) and NOT
    # 'temp_ode.ipynb'. Here is the output:

    # running: ['jupyter', 'nbconvert', '--to', 'notebook',
    #           '--execute', 'ode.ipynb', '--output', 'temp_ode.ipynb']
    #
    # Running Sphinx v5.3.0
    # [NbConvertApp] Converting notebook ode.ipynb to notebook
    # [NbConvertApp] Writing 366060 bytes to temp_ode.nbconvert.ipynb

    if not os.path.exists(tmp):
        tmp = tmp.replace(".ipynb", ".nbconvert.ipynb")

    # Convert to .rst for Sphinx:
    cmd = ["jupyter", "nbconvert", "--to", "rst", tmp, "--output", tmp2]
    print(f"running: {cmd}")
    run(cmd, check=True)

    with open(tmp2) as fin:
        with open(rst, "wt") as fout:
            for line in fin:
                fout.write(line.replace(".. parsed-literal::", ".. code-block:: none"))

    # Remove the temporary notebooks:
    os.remove(tmp)
    os.remove(tmp2)


if __name__ == "__main__":
    for nbname in sys.argv[1:]:
        convert_nb(nbname)
