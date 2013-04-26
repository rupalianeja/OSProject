For checking the execution time of parallel code Vs Sequential code,
Run the bash file in terminal with following command:

bash run sequential.c code_9thapril.cu

After running the command,
Enter the input vector size for sequential code first than the parallel code, which should be same for each iteration.

Next control goes into GNU Plot, there enter the command
load "./time_graph.p"


For checking the SNR Vs BER plot of parallel code Vs Sequential code,
Run the bash file in terminal with following command:

bash run turbo_code2.c beta.cu

After running the command,
Enter the SNR values for sequential code first than the parallel code, which should be same for each iteration.

Next control goes into GNU Plot, there enter the command
load "./graph_ber.p"
