set title "Execution time Sequential Vs Parallel code" font ",20"
set xlabel "Input vector size" font ",15"
set ylabel "Execution time (ms)" font ",15"
set grid
plot "./time_gpu.dat" title "Parallel" lw 3 with lines, "./time_cpu.dat" title "Sequential" lw 3 with lines
