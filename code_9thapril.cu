#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <ctime>

#define MAX_RANDOM 2147483647 
#define NMAX 100000

#define DEBUG     1   //set level of debug visibility [0=>off,1=>min,2=>max]
#define NOISEOFF  0   //set to suppress noise in channel

#define N_ITERATION 4 //no. of turbo decoder iterations

#define N 16000
//#define NA 16000
#define permutationseed 2






#define M  4      //no. of trellis states
int X[N],count=0;
int permutation[N];

int from[M][2];   //from[m][i] = next state (from state m with databit = i)
int to[M][2];     //to[m][i] = previous state (to state m with databit = i)
int parity[M][2]; //parity bit associated with transition from state m
int term[M][2];   //term[m] = pair of data bits required to terminate trellis


void randomInterleaver(){
	

int interleaver[NMAX];
int check[NMAX];                       // Already permuted positions
int i;
int position;


  srandom(permutationseed);

  for (i=0; i<N; i++)
    check[i] = 0;

  for (i=0; i<N; i++)
    {
      do
        position = (int) ( ( (double)(random())/MAX_RANDOM ) * N );
      while ( check[position] );
  

       check[position] = 1;			
      interleaver[i] = position;
    }    

  for (i=0; i<N; i++)
    {
	permutation[i]=interleaver[i];
	X[i]=interleaver[i]%2;
      // printf("%5d -> %5d\n",X[i],permutation[i]);
    	
    }

}



//  Normally distributed number generator (ubiquitous Box-Muller method)
//
double normal(void)
{
	double x, y, rr, randn;
	do{
        x  = (double) 2*rand()/RAND_MAX - 1.0; //uniform in range [-1,1]
        y  = (double) 2*rand()/RAND_MAX - 1.0; //uniform in range [-1,1]
        rr = x*x + y*y;
    } while( rr >= 1 );
    randn = x*sqrt((-2.0*log(rr))/rr);
  return(randn);
}

//  modified BCJR algorithm (MAP decoder)
//
__global__ void calgamma(double *d_gammae,double *d_gamma,int *d_parity,double *d_La,double *d_x_d,double *d_p_d,int Lc)
	{
	int i = blockIdx.x*400+threadIdx.x;
    	int j = blockIdx.y;
	int k = blockIdx.z;	
	double xk_h;
	double pk_h;
	xk_h=k ? +1 : -1;
	pk_h=d_parity[j*2+k] ? +1 : -1;
	d_gamma[M*2*i+2*j+k]=exp(0.5*(d_La[i] * xk_h + Lc * d_x_d[i] * xk_h +
				                          Lc * d_p_d[i] * pk_h));
	d_gammae[M*2*i+2*j+k] = exp(0.5*(Lc * d_p_d[i] * pk_h));

	}


__global__ void calExtLLR(double *d_gammae,double *d_alpha,double *d_beta,int *d_from,double *d_Le)
	{
		int k = blockIdx.x;
		double pr1,pr0;
		pr1=0;
		pr0=0;
		int m;	

		for(m = 0; m < 4; m++)
		{
			//we use gammae rather than gamma as we want the
			//extrinsic component of the overall likelihood
			pr1 += (d_alpha[k*M+m] * d_gammae[k*M*2+m*2+1] * d_beta[(k+1)*M+d_from[m*2+1]]);
			pr0 += (d_alpha[k*M+m] * d_gammae[k*M*2+m*2+0] * d_beta[(k+1)*M+d_from[m*2+0]]);
		}
		d_Le[k] = log(pr1/ pr0); //extrinsic likelihood
	}


/*__global__ void calBeta(double *d_gamma,){
	
	
}*/


__global__ void calAlpha_beta(double *d_gamma,double *d_alpha,int *d_to,double *d_beta, int is_term,int *d_from){
	__shared__ double *dd_gamma;
	__shared__ int *dd_to;
	dd_gamma = d_gamma;
	//dd_alpha = d_alpha;
	dd_to = d_to;
	
	
	double total;
	int k=blockIdx.x*8;
	int q=8200+blockIdx.y*8;
	int a,m;
	
	d_alpha[k+0] = 1;
	for(m = 1; m < M; m++)
		d_alpha[k+m] = 0;

	for(a = 1; a <= 7; a++)
	{
		total = 0;

	    for(m = 0; m < M; m++)
	    {
			d_alpha[a*k+4+m] = d_alpha[(k+(a-1)+4)+dd_to[m*2+0]] * dd_gamma[(k+(a-1)*4*2)+(dd_to[m*2+0]*2)+0] + d_alpha[(k+(a-1)+4)+d_to[m*2+1]] * dd_gamma[(k+(a-1)*4*2)+(dd_to[m*2+1]*2)+1];

			total += d_alpha[a+k+4+m];
		}

		//normalise
		for(m = 0; m < M; m++)
			d_alpha[a+k+4+m] /= total;
	}
	
	
	d_alpha[q+0] = 1;
	for(m = 1; m < M; m++)
		d_alpha[q+m] = 0;

	for(a = 1; a <= 7; a++)
	{
		total = 0;

	    for(m = 0; m < M; m++)
	    {
			d_alpha[a*q+4+m] = d_alpha[(q+(a-1)+4)+dd_to[m*2+0]] * dd_gamma[(q+(a-1)*4*2)+(dd_to[m*2+0]*2)+0] + d_alpha[(q+(a-1)+4)+d_to[m*2+1]] * dd_gamma[(q+(a-1)*4*2)+(dd_to[m*2+1]*2)+1];

			total += d_alpha[a+q+4+m];
		}

		//normalise
		for(m = 0; m < M; m++)
			d_alpha[a+q+4+m] /= total;
	}
	
	
	/* Beta calculation */
	
	
	
	__shared__ int *dd_from;
	
	dd_from = d_from;
	//dd_beta-d_beta;
	
	//int k=blockIdx.x*8;
	int /*a,m,*/f1,f2;
	
	if(is_term)                 //if trellis terminated
	{
		//we know for sure the final state is 0
	    d_beta[k+7+4+0] = 1;
	    for(m = 1; m < M; m++)
	    	d_beta[k+7+4-m] = 0;
	}
	else                       //else trellis not terminated
	{
		//we haven't a clue which is final state
		//so the best we can do is say they're all equally likely
	    for(m = 0; m < M; m++)
	    	d_beta[k+7+4-m] = 1.0 / (double) M;
	}

    //iterate backwards through trellis
	for(a = 6; a >= 0; a--)
	{
		total = 0;
		for(m = 0; m < 4; m++)
		{
			f1=dd_from[m*2+0];
			f2=dd_from[m*2+1];
			d_beta[a+k+4-m] = d_beta[(a+1)+k+4+f1] * dd_gamma[k+a*4*2+m*2+0] +
				         d_beta[(a+1)+k+4+f2] * dd_gamma[k+a*4*2+m*2+1];


			total += d_beta[a+k+4-m];
		}

        //normalise
		for(m = 0; m < 4; m++)
			d_beta[a+k+4-m] /= total;
	}

	if(is_term)                 //if trellis terminated
	{
		//we know for sure the final state is 0
	    d_beta[q+7+4+0] = 1;
	    for(m = 1; m < M; m++)
	    	d_beta[q+7+4-m] = 0;
	}
	else                       //else trellis not terminated
	{
		//we haven't a clue which is final state
		//so the best we can do is say they're all equally likely
	    for(m = 0; m < M; m++)
	    	d_beta[q+7+4-m] = 1.0 / (double) M;
	}

    //iterate backwards through trellis
	for(a = 6; a >= 0; a--)
	{
		total = 0;
		for(m = 0; m < 4; m++)
		{
			f1=dd_from[m*2+0];
			f2=dd_from[m*2+1];
			d_beta[a+q+4-m] = d_beta[(a+1)+q+4+f1] * dd_gamma[q+a*4*2+m*2+0] +
				         d_beta[(a+1)+q+4+f2] * dd_gamma[q+a*4*2+m*2+1];


			total += d_beta[a+q+4-m];
		}

        //normalise
		for(m = 0; m < 4; m++)
			d_beta[a+q+4-m] /= total;
	}


}

void modified_bcjr
(
	int    is_term,      //indicates if trellis terminated
	double Lc,           //Lc = 2/(sigma*sigma) = channel reliability
	double La[N],        //apriori likelihood of each info bit
	double x_d[N],       //noisy data
	double p_d[N],       //noisy parity
	double Le[N]         //extrinsic log likelihood
)
{
	//int    k, m, i;
	//double xk_h, pk_h;      //databit & parity associated with a branch
	double gammae[N][M][2]; //gammas for extrinsic likelihoods
	double gamma[N][M][2];  //gammas for total likelihoods
//	double alpha[N+1][M];   //probability of entering branch via state m
	double beta[N+1][M];    //probability of exiting branch via state m
//	double total;           //used for normalising alpha's and beta's

    
    
    //calculate branch gamma's
	double *d_gammae;
	double *d_gamma;
	int *d_parity;
	double *d_La;
	double *d_x_d;
	double *d_p_d;
	int *d_to;
	int *d_from;
	
	cudaMalloc((void**)&d_gammae,N*M*2*sizeof(double));
	cudaMalloc((void**)&d_gamma,N*M*2*sizeof(double));
	cudaMalloc((void**)&d_parity,M*2*sizeof(int));
	cudaMalloc((void**)&d_La,N*sizeof(double));
	cudaMalloc((void**)&d_x_d,N*sizeof(double));
	cudaMalloc((void**)&d_p_d,N*sizeof(double));
	cudaMalloc((void**)&d_to,M*2*sizeof(int));
	cudaMalloc((void**)&d_from,M*2*sizeof(int));
    	
	cudaMemcpy(d_to,to,M*2*sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy(d_from,from,M*2*sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy(d_parity,parity,M*2*sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy(d_x_d,x_d,N*sizeof(double),cudaMemcpyHostToDevice);
	cudaMemcpy(d_La,La,N*sizeof(double),cudaMemcpyHostToDevice);
	cudaMemcpy(d_p_d,p_d,N*sizeof(double),cudaMemcpyHostToDevice);
    
	dim3 grid(N/400,M,2);

	calgamma<<<grid,400>>>(d_gammae,d_gamma,d_parity,d_La,d_x_d,d_p_d,Lc);
	cudaMemcpy(gamma,d_gamma,M*N*2*sizeof(double),cudaMemcpyDeviceToHost);
	cudaMemcpy(gammae,d_gammae,M*N*2*sizeof(double),cudaMemcpyDeviceToHost);
	
	
	
		
	//cudaFree(d_gamma);
	cudaFree(d_parity);
	cudaFree(d_La);
	cudaFree(d_x_d);
	cudaFree(d_p_d);


	



	//  Calculate state alpha's
	//
   
	double *d_alpha;
	double *d_beta;
	cudaMalloc((void**)&d_beta,(N+1)*M*sizeof(double));
	
	cudaMalloc((void**)&d_alpha,(N+1)*M*sizeof(double));
	
	dim3 grid1(1024,1024);
	
	calAlpha_beta<<<grid1,1>>>(d_gamma,d_alpha,d_to,d_beta,is_term,d_from);
	
	
	cudaMemcpy(gamma,d_gamma,(N+1)*M*sizeof(double),cudaMemcpyDeviceToHost);
	cudaMemcpy(beta,d_beta,(N+1)*M*sizeof(double),cudaMemcpyDeviceToHost);
	//  Calculate state beta's
	//
   
	


    //  Calculate extrinsic likelihood
    //
	


	//double *d_alpha;
	
	//int *d_from;
	double *d_Le;
	
	
	//cudaMalloc((void**)&d_alpha,(N+1)*M*sizeof(double));
	
	//cudaMalloc((void**)&d_from,M*2*sizeof(int));
	cudaMalloc((void**)&d_Le,N*sizeof(double));
	
	
	//cudaMemcpy(d_alpha,alpha,(N+1)*M*sizeof(double),cudaMemcpyHostToDevice);
	//cudaMemcpy(d_beta,beta,(N+1)*M*sizeof(double),cudaMemcpyHostToDevice);
	cudaMemcpy(d_from,from,M*2*sizeof(int),cudaMemcpyHostToDevice);
    

	calExtLLR<<<N,1>>>(d_gammae,d_alpha,d_beta,d_from,d_Le);

	
	cudaMemcpy(Le,d_Le,N*sizeof(double),cudaMemcpyDeviceToHost);

	cudaFree(d_gammae);
	cudaFree(d_alpha);
	cudaFree(d_beta);
	cudaFree(d_Le);



    #if DEBUG > 1
    for(k = 0; k < N; k++)
    {
		for(m = 0; m < M; m++)
		{
			for(i = 0; i < 2; i++)
			{
				printf("gamma[%i][%i][%i]  = %f\t", k, m, i, gamma[k][m][i]);
				printf("gammae[%i][%i][%i] = %f\n", k, m, i, gammae[k][m][i]);
			}
		}
		printf("\n");
	}

	for(k = 0; k <= N; k++)
	{
	    for(m = 0; m < M; m++)
			printf("alpha[%i][%i] = %f\n", k, m, alpha[k][m]);
		printf("\n");
	}
	for(k = 0; k <= N; k++)
	{
	    for(m = 0; m < M; m++)
			printf("beta[%i][%i] = %f\n", k, m, beta[k][m]);
		printf("\n");
	}
    #endif

}

//
//      +--------------------------> Xk
//      |  fb
//      |  +---------(+)-------+
//      |  |          |        |
//  Xk--+-(+)-+->[D]----->[D]--+
//            |                |
//            +--------------(+)---> Pk
//
//
void gen_tab(void)
{
	int m, i, b0, b1, fb, state;

    //generate tables for 4 state RSC encoder
	for(m = 0; m < M; m++) //for each starting state
		for(i = 0; i < 2; i++) //for each possible databit
		{
			b0 = (m >> 0) & 1; //bit 0 of state
			b1 = (m >> 1) & 1; //bit 1 of state

			//parity from state m with databit i
			parity[m][i] = b0 ^ i;

			//from[m][i] = next state from state m with databit i
			from[m][i]   = (b0<<1) + (i ^ b0 ^ b1);
		}

    //to[m][i] = previous state to state m with databit i
    for(m = 0; m < M; m++)
    	for(i = 0; i < 2; i++)
			to[from[m][i]][i] = m;

	
	for(m = 0; m < M; m++) //for each state
	{
		state = m;
		b0 = (state >> 0) & 1; //bit 0 of state
		b1 = (state >> 1) & 1; //bit 1 of state
		fb = b0 ^ b1;          //feedback bit
		term[m][0] = fb;       //will set X[N-2] = fb

		state = from[m][fb];   //advance from state m with databit=fb
		b0 = (state >> 0) & 1; //bit 0 of state
		b1 = (state >> 1) & 1; //bit 1 of state
		fb = b0 ^ b1;          //feedback bit
		term[m][1] = fb;       //will set X[N-1] = fb
	}
}

//
//       +-----------> Xk
//       |
//       |
//       |
//  Xk---+--[E1]-----> P1k
//       |
//      [P]
//       |
//       +--[E2]-----> P2k
//
//
void turbo_encode
(
	int X[N],   //block of N-2 information bits + 2 to_be_decided bits
	int P1[N],  //encoder #1 parity bits
	int P2[N]   //encoder #2 parity bits
)
{
	int    k;      //trellis stage
	int    state;  //encoder state
	int    X_p[N]; //X_permuted = permuted bits

	//encoder #1
	state = 0; //encoder always starts in state 0
	for(k = 0; k < N-2; k++)
	{
		P1[k] = parity[state][X[k]];
		state = from[state][X[k]];
		//printf("s[%i] = %i\n", k, state);
	}

	//terminate encoder #1 trellis to state 0
	X[N-2]  = term[state][0];  //databit to feed a 0 into delay line
	X[N-1]  = term[state][1];  //databit to feed another 0 into delay line

	P1[N-2] = parity[state][X[N-2]]; //parity from state with databitX[N-2]
	state   = from[state][X[N-2]];   //next state from current state
    P1[N-1] = parity[state][X[N-1]]; //parity from state with databit=X[N-1]
	state   = from[state][X[N-1]];   //next state from current state

	if(state != 0)
	{
		//should never get here
		printf("Error: Could not terminate encoder #1 trellis\n");
		exit(1);
	}

	//permute tx databits for encoder #2
	for(k = 0; k < N; k++)
		X_p[k] = X[permutation[k]];

	//encoder #2
	state = 0; //encoder always starts in state 0
	for(k = 0; k < N; k++)
	{
		P2[k] = parity[state][X_p[k]]; //parity from state with databit=X_p[k]
		state = from[state][X_p[k]];   //next state from current state
	}

	//for(k = 0; k < N; k++)
	//	printf("%i %i %i %i\n", X[k], P1[k], X_p[k], P2[k]);

}

void turbo_decode(
	int NA,
	double sigma,   //channel noise standard deviation
	double x_d[N],  //x_dash  = noisy data symbol
	double p1_d[N], //p1_dash = noisy parity#1 symbol
	double p2_d[N], //p2_dash = noisy parity#2 symbol
	double L_h[N],  //L_hat = likelihood of databit given entire observation
	int X_h[]	
)
{
	int i, k;

	double Le1[N];    //decoder #1 extrinsic likelihood
	double Le1_p[N];  //decoder #1 extrinsic likelihood permuted
	double Le2[N];    //decoder #2 extrinsic likelihood
	double Le2_ip[N]; //decoder #2 extrinsic likelihood inverse permuted
    double Lc;        //channel reliability value

    Lc = 2.0 / (sigma*sigma); //requires sigma to be non-trivial

    //zero apriori information into very first iteration of BCJR
    for(k = 0; k < N; k++)
		Le2_ip[k] = 0;

    for(i = 0; i < N_ITERATION; i++)
    {
    	modified_bcjr(1, Lc, Le2_ip, x_d, p1_d, Le1);

        //permute decoder#1 likelihoods to match decoder#2 order
    	for(k = 0; k < N; k++)
    		Le1_p[k] = Le1[permutation[k]];

    	modified_bcjr(0, Lc, Le1_p,  x_d, p2_d, Le2);

        //inverse permute decoder#2 likelihoods to match decoder#1 order
    	for(k = 0; k < N; k++)
    		Le2_ip[permutation[k]] = Le2[k];

        #if DEBUG > 1
		for(k = 0; k < N; k++)
		{
 			printf("Le1[%i] = %f\t", k, Le1[k]);
 			printf("Le2_ip[%i] = %f\t", k, Le2_ip[k]);
 			//printf("Lc*x_d[%i] = %f", k, Lc*x_d[k]);
			printf("\n");
		}
		printf("\n");
		#endif
	}

    //calculate overall likelihoods and then slice'em
    for(k = 0; k < N; k++)
    {
		L_h[k] = Lc*x_d[k] + Le1[k] + Le2_ip[k]; //soft decision
		X_h[count] = (L_h[k] > 0.0) ? 1 : 0;
		count++;         //hard decision
	}
}

/*
gcc turbo_example.c -lm -o t; t
*/

int main(void)
{
    
	int NA;
	printf("Enter the NA value");
	scanf("%d",&NA);

	float snr;
	int snrdb;
	double noise;
	int k,i;         //databit index (trellis stage)
	int signal_power=1;	
	double sigma;
	int    P1[N];     //encoder #1 parity bits
	int    P2[N];     //encoder #2 parity bits
	double x[N];      //databit mapped to symbol
	double p1[N];     //encoder #1 parity bit mapped to symbol
	double p2[N];     //encoder #2 parity bit mapped to symbol
	double x_d[N];    //x_dash  = noisy data symbol
	double p1_d[N];   //p1_dash = noisy parity#1 symbol
	double p2_d[N];   //p2_dash = noisy parity#2 symbol
	double L_h[N];    //L_hat = likelihood of databit given entire observation
	//int    X_h[N];    //X_hat = sliced MAP decisions
	double elapsed;
    	clock_t t1, t2;
	double time_count;
	int input[NA],X_h[NA];
	
	
	
	FILE *fp= fopen("time_gpu.dat","a+");
	for(i=0; i< NA; i++){
		X_h[i]=0;
	}
	
	/*printf("Enter the SNR value in db");
	scanf("%d",&snrdb);*/
	snrdb=5;
	snr= pow(10,(float)snrdb/10);
	
	noise = (float)signal_power/snr;
	
	printf("signal power is %d \n",signal_power);
	for(i=0;i < (NA/N);i++){
    		randomInterleaver();
		for(k=0;k<N;k++){
			input[k+i*N]= X[k];
		}
		
	
		

	srand(1);    //init random number generator
	gen_tab();   //generate trellis tables
	sigma  = sqrt(noise);  //noise std deviation

    /********************************
     *           ENCODER            *
     ********************************/

    turbo_encode(X, P1, P2);

    //map bits to symbols
	for(k = 0; k < N; k++) //for entire block length
	{
		x[k]  = X[k]  ? +1 : -1;  //map databit to symbol
		p1[k] = P1[k] ? +1 : -1;  //map parity #1 to symbol
		p2[k] = P2[k] ? +1 : -1;  //map parity #2 to symbol
	}

    /********************************
     *           CHANNEL            *
     ********************************/

    //add some AWGN
	for(k = 0; k < N; k++)
	{
		#if NOISEOFF
		x_d[k]  = x[k];
		p1_d[k] = p1[k];
		p2_d[k] = p2[k];
		#else
		x_d[k]  = x[k]  + sigma*normal();
		p1_d[k] = p1[k] + sigma*normal();
		p2_d[k] = p2[k] + sigma*normal();
		#endif
	}

    #if DEBUG > 1
	for(k = 0; k < N; k++)
		printf("%f \t%f \t%f\n", x_d[k], p1_d[k], p2_d[k]);
	#endif

    /********************************
     *           DECODER            *
     ********************************/

	t1 = clock();
	turbo_decode(NA,sigma, x_d, p1_d, p2_d, L_h, X_h);
	t2=clock();
	time_count=time_count+((double)t2-(double)t1);
	}
	/*printf("\n\n****INPUT****\n\n");
	for(i=0;i < NA;i++){
		printf("%d",input[i]);
	}*/
	
	/*printf("\n\n****OUTPUT****\n\n");

	printf("X_h = ");
	for(k = 0; k < NA; k++)
	    	printf("%i", X_h[k]);
	printf("\n");*/
	
	int count=0;
	//float ber;
	for(k=0; k < NA; k++) {
		if(X_h[k] != input[k])
		count++;
	}
	//ber=(float)count/NA;
	//printf("BER is %f",ber);
	//printf("count is %d",count);
	elapsed = time_count / CLOCKS_PER_SEC * 1000;
         fprintf(fp,"%d %lf",NA, elapsed);
         fprintf(fp,"\n");
    	fclose(fp);
	
	printf("\nTime elapsed =%lf ms\n",elapsed);
    
    	return 0;

}

