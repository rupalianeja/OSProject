
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define MAX_RANDOM 2147483647 
#define NMAX 16384

#define DEBUG     1   //set level of debug visibility [0=>off,1=>min,2=>max]
#define NOISEOFF  0   //set to suppress noise in channel

#define N_ITERATION 4 //no. of turbo decoder iterations
#define TBD -1        //trellis termination bits (inserted by encoder #1)


//#define NA 1600000
#define N    16000
#define permutationseed 3



#define M  4      //no. of trellis states
int X[N],count=0;
//int X_h[NA];
int X[N];
int permutation[N];
int from[M][2];   //from[m][i] = next state (from state m with databit = i)
int to[M][2];     //to[m][i] = previous state (to state m with databit = i)
int parity[M][2]; //parity bit associated with transition from state m
int term[M][2];   //term[m] = pair of data bits required to terminate trellis
//int inputbit[NA];
int counter=0;

//  Normally distributed number generator (ubiquitous Box-Muller method)
//


void randomInterleaver(){

int interleaver[NMAX];
int check[NMAX];                       // Already permuted positions
int i;
int position;


  srandom(permutationseed);

  for (i=0; i<N; i++)
    check[i] = 0;

  for (i=0; (i<N); i++)
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
//       printf("%5d -> %5d\n",X[i],permutation[i]);
    	
    }
counter = counter + 16000;

}

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
	int    k, m, i;
	double xk_h, pk_h;      //databit & parity associated with a branch
	double gamma[N][M][2];  //gammas for total likelihoods
	double gammae[N][M][2]; //gammas for extrinsic likelihoods
	double pr0, pr1;        //extrinsic likelihood = pr1/pr0
	double alpha[N+1][M];   //probability of entering branch via state m
	double beta[N+1][M];    //probability of exiting branch via state m
	double total;           //used for normalising alpha's and beta's

    //calculate branch gamma's
    for(k = 0; k < N; k++) //for each trellis stage
    {
		for(m = 0; m < M; m++) //for each state
		{
			for(i = 0; i < 2; i++) //for each databit
			{
				//data associated with branch
				xk_h = i ? +1 : -1;            //map databit to PAM symbol

				//parity associated with branch
				pk_h = parity[m][i] ? +1 : -1; //map parity bit to PAM symbol

                //used later to calculate alpha's and beta's
				gamma[k][m][i] = exp(0.5*(La[k] * xk_h +
				                          Lc * x_d[k] * xk_h +
				                          Lc * p_d[k] * pk_h));

                //used later to calculate extrinsic likelihood
				gammae[k][m][i] = exp(0.5*(Lc * p_d[k] * pk_h));
			}
		}
	}

	//  Calculate state alpha's

	alpha[0][0] = 1;
	for(m = 1; m < M; m++)
		alpha[0][m] = 0;

	for(k = 1; k <= N; k++)
	{
		total = 0;

	    for(m = 0; m < M; m++)
	    {
			alpha[k][m] = alpha[k-1][to[m][0]] * gamma[k-1][to[m][0]][0] +
			              alpha[k-1][to[m][1]] * gamma[k-1][to[m][1]][1];

			total += alpha[k][m];
		}

		//normalise
		for(m = 0; m < M; m++)
			alpha[k][m] /= total;
	}

	//  Calculate state beta's


	if(is_term)                 //if trellis terminated
	{
		//we know for sure the final state is 0
	    beta[N][0] = 1;
	    for(m = 1; m < M; m++)
	    	beta[N][m] = 0;
	}
	else                       //else trellis not terminated
	{
		//we haven't a clue which is final state
		//so the best we can do is say they're all equally likely
	    for(m = 0; m < M; m++)
	    	beta[N][m] = 1.0 / (double) M;
	}

    //iterate backwards through trellis
	for(k = N-1; k >= 0; k--)
	{
		total = 0;
		for(m = 0; m < 4; m++)
		{
			beta[k][m] = beta[k+1][from[m][0]] * gamma[k][m][0] +
				         beta[k+1][from[m][1]] * gamma[k][m][1];


			total += beta[k][m];
		}

        //normalise
		for(m = 0; m < 4; m++)
			beta[k][m] /= total;
	}

    //  Calculate extrinsic likelihood
    //
	for(k = 0; k < N; k++)
	{
		pr0 = pr1 = 0;
		for(m = 0; m < 4; m++)
		{
			//we use gammae rather than gamma as we want the
			//extrinsic component of the overall likelihood
			pr1 += (alpha[k][m] * gammae[k][m][1] * beta[k+1][from[m][1]]);
			pr0 += (alpha[k][m] * gammae[k][m][0] * beta[k+1][from[m][0]]);
		}
		Le[k] = log(pr1 / pr0); //extrinsic likelihood
	}

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

        #if DEBUG > 0
		/*for(k = 0; k < N; k++)
		{
 			printf("Le1[%i] = %f\t", k, Le1[k]);
 			printf("Le2_ip[%i] = %f\t", k, Le2_ip[k]);
 			//printf("Lc*x_d[%i] = %f", k, Lc*x_d[k]);
			printf("\n");
		}
		printf("\n");*/
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
	printf("Enter NA");
	scanf("%d",&NA);
	int k,i;  
	char ch;
   	//FILE *fp=fopen("quantbit.dat","r");
   	FILE *fpout=fopen("time_cpu.dat","a+");
	//fscanf(fp,"%c",&ch);
	printf("-----------input bits----------\n");
   
	int snr;
	int snrdb;
	double noise;
	       //databit index (trellis stage)
	int signal_power=1;	
	double sigma = 1.0;
	int    P1[N];     //encoder #1 parity bits
	int    P2[N];     //encoder #2 parity bits
	double x[N];      //databit mapped to symbol
	double p1[N];     //encoder #1 parity bit mapped to symbol
	double p2[N];     //encoder #2 parity bit mapped to symbol
	double x_d[N];    //x_dash  = noisy data symbol
	double p1_d[N];   //p1_dash = noisy parity#1 symbol
	double p2_d[N];   //p2_dash = noisy parity#2 symbol
	double L_h[N];    //L_hat = likelihood of databit given entire observation
	//int    X_h[NA];    //X_hat = sliced MAP decisions
	double elapsed;
    	clock_t t1, t2;
	double time_count;
	int input[NA], X_h[NA];
	//FILE *fp= fopen("ber.dat","a+");
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
	turbo_decode(NA,sigma, x_d, p1_d, p2_d, L_h,X_h);
	t2=clock();
	time_count=time_count+((double)t2-(double)t1);
	}
	
	
	int count1=0;
	float ber;
	for(k=0; k < NA; k++) {
		if(X_h[k] != input[k])
		count1++;
	}
	ber=(float)count1/NA;
	//printf("BER is %f",ber);
	//printf("count is %d",count1);
         /*fprintf(fp,"%d %f",snrdb, ber);
         fprintf(fp,"\n");*/
    
	elapsed = time_count / CLOCKS_PER_SEC * 1000;
	printf("\n\n Time elapsed =%lf ms\n",elapsed);
	
	/*for(i=0; i< NA;i++){
		fprintf(fpout,"%d",X_h[i]);
		fprintf(fpout,"\n");
	}*/
	
	fprintf(fpout,"%d %lf",NA,elapsed);
	fprintf(fpout,"\n");
	
    	//fclose(fp);
    	fclose(fpout);
    	return 0;

}

