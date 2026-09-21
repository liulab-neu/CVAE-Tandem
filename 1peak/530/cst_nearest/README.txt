Nearest-neighbour comparison, not a simulation of the design.

plot_cst_nearest_single.py does not run CST on the predicted structure.  It
searches the 1312-sample CST dataset for the angle-averaged spectrum with the
smallest full-spectrum MSE against the CVAE-assisted network response, and
reports that spectrum.  The structure it belongs to is therefore close to, but
not equal to, the design.

../cst/ holds the other kind of check: CST run on the predicted structure
itself, at 0, 30 and 60 degrees.  The two directories use the same file names,
so keep them apart -- the script writes to whichever directory it is pointed
at, and running it against ../cst/ overwrites the real simulation.
