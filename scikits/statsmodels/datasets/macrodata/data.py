"""United States Macroeconomic data"""

__all__ = ['COPYRIGHT','TITLE','SOURCE','DESCRSHORT','DESCRLONG','NOTE', 'load']


__docformat__ = 'restructuredtext'

COPYRIGHT   = """This is public domain."""
TITLE       = __doc__
SOURCE      = """
Compiled by Skipper Seabold. All data are from the Federal Reserve Bank of St.
Louis [1] except the unemployment rate which was taken from the National
Bureau of Labor Statistics [2]. ::

    [1] Data Source: FRED, Federal Reserve Economic Data, Federal Reserve Bank of
        St. Louis; http://research.stlouisfed.org/fred2/; accessed December 15,
        2009.

    [2] Data Source: Bureau of Labor Statistics, U.S. Department of Labor;
        http://www.bls.gov/data/; accessed December 15, 2009.
"""

DESCRSHORT  = """US Macroeconomic Data for 1959Q1 - 2009Q3"""

DESCRLONG   = DESCRSHORT

NOTE        = """
Number of Observations - 203

Number of Variables - 14

Variable name definitions::

    year      - 1959q1 - 2009q3
    quarter   - 1-4
    realgdp   - Real gross domestic product (Bil. of chained 2005 US$,
                seasonally adjusted annual rate)
    realcons  - Real personal consumption expenditures (Bil. of chained 2005
                US$,
                seasonally adjusted annual rate)
    realinv   - Real gross private domestic investment (Bil. of chained 2005
                US$, seasonally adjusted annual rate)
    realgovt  - Real federal consumption expenditures & gross investment
                (Bil. of chained 2005 US$, seasonally adjusted annual rate)
    realdpi   - Real gross private domestic investment (Bil. of chained 2005
                US$, seasonally adjusted annual rate)
    cpi       - End of the quarter consumer price index for all urban
                consumers: all items (1982-84 = 100, seasonally adjusted).
    m1        - End of the quarter M1 nominal money stock (Seasonally adjusted)
    tbilrate  - Quarterly monthly average of the monthly 3-month treasury bill:
                secondary market rate
    unemp     - Seasonally adjusted unemployment rate (%)
    pop       - End of the quarter total population: all ages incl. armed
                forces over seas
    infl      - Inflation rate (ln(cpi_{t}/cpi_{t-1}) * 400)
    realint   - Real interest rate (tbilrate - infl)
"""

from numpy import recfromtxt, column_stack, array
from scikits.statsmodels.datasets import Dataset
from os.path import dirname, abspath

def load():
    """
    Load the US macro data and return a Dataset class.

    Returns
    -------
    Dataset instance:
        See DATASET_PROPOSAL.txt for more information.

    Notes
    -----
    The macrodata Dataset instance does not contain endog and exog attributes.
    """
    filepath = dirname(abspath(__file__))
    data = recfromtxt(open(filepath + '/macrodata.csv', 'rb'), delimiter=",", names=True,
            dtype=float)
    names = data.dtype.names
    dataset = Dataset(data=data, names=names)
    return dataset
