In this project, I re-code some of the exercises in Ng A.'s Machine Learing course, using Python instead of original OCTAVE/MATLAB.

Menu:
1. ex2--Logistic Regression

Details:
1. ex2
    * Plot the data.
        * Load csv data using method csv.reader().
            * Set argument "quoting" to csv.QUOTE_NONNUMERIC/QUOTE_NONE so that data is loaded as floats.
        * Put data in ndarray, instead of list.
        * Use pandas/matplotlib to plot the data.
            * Create pd.DataFrame objects.
            * Use plot.scatter() method to draw a scatter diagram (accept a pd.DataFrame object as a argument).
            * NOTE: pandas can integrate lines of matplotlib commands into a one method.
    * Use "Nelder-Mead" and "BFGS" methods to minimize cost function.
    Finally we get the same result as OCTAVE version gave.
