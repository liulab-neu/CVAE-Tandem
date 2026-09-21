Training logs of the checkpoints in ../model, copied out of the weight sweep.

Each network was trained once per cell of the (w_peak-wavelength, w_peak-intensity)
grid and the cell with the smallest mean dip-position error was kept, so the three
networks do not share a cell and their logs come from three different runs:

  DNN_FNN-t.txt      forward   w_peak-wavelength=1    w_peak-intensity=1
  DNN_tandem-t.txt   tandem    w_peak-wavelength=10   w_peak-intensity=1
  CVAE_2p-t.txt      CVAE      w_peak-wavelength=10   w_peak-intensity=100

The logs directly under 2peak/ belong to a different run and do not match the
deployed checkpoints; these are the ones the loss figure plots.
