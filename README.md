In this project, I re-code some of the exercises in Ng A.'s Machine Learing course, using Python instead of original OCTAVE/MATLAB.

Menu:
1. ex2--Logistic Regression

Details:
1. ex2
    * plot the data
        * load csv data using method csv.reader()
            * set argument "quoting" to csv.QUOTE_NONNUMERIC/QUOTE_NONE so that data is loaded as floats
        * put data in np.array, instead of list
        * use pandas/matplotlib to plot the data
            * create pd.DataFrame objects
            * use plot.scatter() method to draw a scatter diagram (accept a pd.DataFrame object as a argument)
            * NOTE: pandas can integrate lines of matplotlib commands into a one method
