set title "SNR Vs BER Sequential Vs Parallel code" font ",10"
set xlabel "SNR (db)" font ",10"
set ylabel "BER" font ",10"
set logscale y
set grid
plot "./bersnr4.dat" title "Sequential" lw 1 with lines, "./snrber2.dat" title "Parallel" lw 1 with lines
