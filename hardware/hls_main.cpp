#include "ap_int.h"

typedef ap_uint<64> uint_t ; //have to change every uint_t to double 

const int SIZE = 6;
const int  D_MODEL = 6 ;
const int C_SIZE =4 ;  //used in q,k, v weights matrix init 
struct transformer {
    char words[SIZE][10];
    int numbers[SIZE];
    double values[SIZE][SIZE];   //initial value matrix 6*6
    double pe[SIZE][SIZE];         //positional encoding matrix 6*6
    double ep[SIZE][SIZE] ;        //total input (pe + values) encoding matrix 6*6
    double ep_trans[SIZE][SIZE] ;        //transpose matrix of ep total input (pe + values) encoding matrix 6*6
    double linear_weights[C_SIZE][SIZE] ;  //linear weight matrix of 4*6
    double weights_for_linear_layer1[SIZE][SIZE] ;//considering 1 liner layer 
    double bias_layer1[1][SIZE] ; //bias matrix for linear [layer 1] 1*6

    double q_weights[SIZE][C_SIZE];   //query weights be 6*4
    double k_weights[SIZE][C_SIZE];   //key weights be 6*4
    double v_weights[SIZE][C_SIZE]; //value weights be 6*4
    double q[SIZE][C_SIZE] ;       // q matrix 
    double k[SIZE][C_SIZE] ;        // k matrix 
    double v[SIZE][C_SIZE] ;         // v matrix 
    double qk_trans[SIZE][SIZE] ;  //q*k(transpose)  it is 6*6
    double qkv_[SIZE][C_SIZE] ;   // [q*k(t)/sqrt(dk)]*V  it is 6*4 
   
    double resultant_qkv[SIZE][SIZE] ; //weights*attention filter  6*6
    double add_qkv_ep[SIZE][SIZE] ; //addition of ep + attention filter  6*6
    double normalized[SIZE][SIZE] ; //normalized*attention filter  6*6

    double linear1_output[SIZE][SIZE] ;  //output of linear layer 1 
    double relu1_output[SIZE][SIZE] ;    //output of activation function relu 1 
    double add_encoder2[SIZE][SIZE] ;   //addition of relu and normalizing  6*6
    double normalized_2enc[SIZE][SIZE] ; //normalized final output of encoder    6*6
};
/////////////////////////////////////////////HLS CODE ?????????????????????????
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ap_fixed.h"
#include "hls_math.h"
#define SIZE 6
#define C_SIZE 4

const double EPSILON = 0.0001; // Small constant
double mean[SIZE] ;
double std_dev[SIZE] ;

typedef ap_fixed<8, 2> data_t;


////////////////////////////////////////NORMALISATION LAYER///////////////////////////////////////////////////////////////////////////
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ap_fixed.h"
#include "hls_math.h"
#define SIZE 6
#define C_SIZE 4

const double EPSILON = 0.0001; // Small constant
double mean[SIZE] ;
double std_dev[SIZE] ;

typedef ap_fixed<8, 2> data_t;


////////////////////////////////////////NORMALISATION LAYER///////////////////////////////////////////////////////////////////////////
void normalize_matrix(data_t add_qkv_ep_buff[SIZE][SIZE], data_t resultant_qkv_buff[SIZE][SIZE], data_t normalized_buff[SIZE][SIZE]) {
    data_t mean[SIZE];
    data_t std_dev[SIZE];

    // Calculate mean and standard deviation
    for (int i = 0; i < SIZE; i++) {
#pragma HLS PIPELINE
        data_t sum = 0.0f;
        data_t variance_sum = 0.0f;

        // First pass: calculate mean
        for (int j = 0; j < SIZE; j++) {
            sum += resultant_qkv_buff[i][j];
        }
        mean[i] = sum / SIZE;

        // Second pass: calculate variance
        for (int j = 0; j < SIZE; j++) {
            data_t diff = resultant_qkv_buff[i][j] - mean[i];
            variance_sum += diff * diff;
        }
        std_dev[i] = hls::sqrt(variance_sum / SIZE);
    }

    // Normalize the matrix
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
#pragma HLS PIPELINE
            normalized_buff[i][j] = (add_qkv_ep_buff[i][j] - mean[i]) / (std_dev[i] + EPSILON);
        }
    }
}
////////////////////////////////////////RELU LAYER///////////////////////////////////////////////////////////////////////////
void relu(
    data_t relu1_output[SIZE][SIZE],
    data_t linear1_output[SIZE][SIZE]) {
#pragma HLS INLINE off
#pragma HLS PIPELINE

    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
#pragma HLS PIPELINE
            relu1_output[i][j] = (linear1_output[i][j] > data_t(0)) ? linear1_output[i][j] : data_t(0);
        }
    }
}


////////////////////////////////////////SOFTMAX LAYER///////////////////////////////////////////////////////////////////////////
void softmax(data_t qk_trans[SIZE][SIZE]) {

    for (int i = 0; i < SIZE; i++) {
        data_t row_sum = 0.0;

        // Calculate sum of exp(x) for the denominator
        for (int j = 0; j < SIZE; j++) {
#pragma HLS PIPELINE
            row_sum += hls::exp(qk_trans[i][j]);
        }

        // Apply softmax formula to each element in the row
        for (int j = 0; j < SIZE; j++) {
#pragma HLS PIPELINE
            qk_trans[i][j] =  hls::exp(qk_trans[i][j]) / row_sum;

        }
    }
}

////////////////////////////////////////LINEAR LAYER///////////////////////////////////////////////////////////////////////////

void linear_layer1(data_t normalized[SIZE][SIZE],
		data_t weights_for_linear_layer1[SIZE][SIZE],
		data_t bias_layer1[SIZE],
		data_t relu1_output[SIZE][SIZE],
		data_t linear1_output[SIZE][SIZE]) {

    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
#pragma HLS PIPELINE
        	linear1_output[i][j] = 0;
        }
    }

    // Matrix multiplication and bias addition
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
#pragma HLS PIPELINE
            for (int k = 0; k < SIZE; k++) {
            	linear1_output[i][j] += normalized[i][k] * weights_for_linear_layer1[k][j];
            }
            linear1_output[i][j] += bias_layer1[j];
        }
    }
    // Apply ReLU activation
       relu(relu1_output, linear1_output);
}
////////////////////////////////////////ADD LAYER///////////////////////////////////////////////////////////////////////////
void addition(
    data_t ep_trans[SIZE][SIZE],
    data_t resultant_qkv[SIZE][SIZE],
    data_t add_qkv_ep[SIZE][SIZE]) {
#pragma HLS INLINE off

    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
#pragma HLS PIPELINE
            add_qkv_ep[i][j] = ep_trans[i][j] + resultant_qkv[i][j];
        }
    }
}
////////////////////////////////////////TRANSPOSE LAYER///////////////////////////////////////////////////////////////////////////
void transpose(
    data_t ep_trans[SIZE][SIZE],
    data_t ep[SIZE][SIZE]) {
#pragma HLS INLINE off

    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
#pragma HLS PIPELINE
            ep_trans[j][i] = ep[i][j];
        }
    }
}


////////////////////////////////////////ATTENTION BLOCK///////////////////////////////////////////////////////////////////////////
//
//void qkv_finding(data_t ep_trans_buff[SIZE][SIZE], data_t q_weights_buff[SIZE][C_SIZE], data_t k_weights_buff[SIZE][C_SIZE], data_t v_weights_buff[SIZE][C_SIZE], data_t q_buff[SIZE][C_SIZE], data_t k_buff[SIZE][C_SIZE], data_t v_buff[SIZE][C_SIZE]) {
//    // Initialize Q, K, V matrices
//    for (int i = 0; i < SIZE; i++) {
//        for (int j = 0; j < C_SIZE; j++) {
//#pragma HLS PIPELINE
//            q_buff[i][j] = 0;
//            k_buff[i][j] = 0;
//            v_buff[i][j] = 0;
//        }
//    }
//
//    // Compute Q matrix
//    for (int i = 0; i < SIZE; i++) {
//        for (int j = 0; j < C_SIZE; j++) {
//#pragma HLS PIPELINE
//            for (int k = 0; k < SIZE; k++) {
//                q_buff[i][j] += ep_trans_buff[i][k] * q_weights_buff[k][j];
//            }
//        }
//    }
//
//    // Compute K matrix
//    for (int i = 0; i < SIZE; i++) {
//        for (int j = 0; j < C_SIZE; j++) {
//#pragma HLS PIPELINE
//            for (int k = 0; k < SIZE; k++) {
//                k_buff[i][j] += ep_trans_buff[i][k] * k_weights_buff[k][j];
//            }
//        }
//    }
//
//    // Compute V matrix
//    for (int i = 0; i < SIZE; i++) {
//        for (int j = 0; j < C_SIZE; j++) {
//#pragma HLS PIPELINE
//            for (int k = 0; k < SIZE; k++) {
//                v_buff[i][j] += ep_trans_buff[i][k] * v_weights_buff[k][j];
//            }
//        }
//    }
//}
//
//void qk_trans(data_t q_buff[SIZE][C_SIZE], data_t k_buff[C_SIZE][SIZE], data_t qk_trans_buff[SIZE][SIZE]) {
//    // Initialize qk_trans_buff
//    for (int i = 0; i < SIZE; i++) {
//        for (int j = 0; j < SIZE; j++) {
//#pragma HLS PIPELINE
//            qk_trans_buff[i][j] = 0;
//        }
//    }
//
//    // Perform matrix multiplication: q * k^T
//    for (int i = 0; i < SIZE; i++) {
//        for (int j = 0; j < SIZE; j++) {
//#pragma HLS PIPELINE
//            for (int k = 0; k < C_SIZE; k++) {
//                qk_trans_buff[i][j] += q_buff[i][k] * k_buff[j][k] / hls::sqrt(D_MODEL);  // Note: k[j][k] instead of k[k][j] for transpose
//            }
//        }
//    }
//}
//
//
//
//
//
//
//
//void qkv_final(data_t qkv_buff[SIZE][C_SIZE], data_t qk_trans_buff[SIZE][SIZE], data_t v_buff[SIZE][C_SIZE], data_t qkv_result_buff[SIZE][C_SIZE]) {
//    // Initialize qkv_result_buff
//    for (int i = 0; i < SIZE; i++) {
//        for (int j = 0; j < C_SIZE; j++) {
//#pragma HLS PIPELINE
//            qkv_result_buff[i][j] = 0;
//        }
//    }
//
//    // Perform matrix multiplication: qk*V
//    for (int i = 0; i < SIZE; i++) {
//        for (int j = 0; j < C_SIZE; j++) {
//#pragma HLS PIPELINE
//            for (int k = 0; k < SIZE; k++) {
//                qkv_result_buff[i][j] += qk_trans_buff[i][k] * v_buff[k][j];
//            }
//        }
//    }
//}
//
//
//
//
//
//
//
//
//void resultant_qkv(data_t qkv_buff[SIZE][C_SIZE], data_t linear_weights_buff[C_SIZE][SIZE], data_t resultant_qkv_buff[SIZE][SIZE]) {
//    // Initialize resultant_qkv_buff
//    for (int i = 0; i < SIZE; i++) {
//        for (int j = 0; j < SIZE; j++) {
//#pragma HLS PIPELINE
//            resultant_qkv_buff[i][j] = 0;
//        }
//    }
//
//    // Perform matrix multiplication: qk*V
//    for (int i = 0; i < SIZE; i++) {
//        for (int j = 0; j < SIZE; j++) {
//#pragma HLS PIPELINE
//            float sum = 0;
//            for (int k = 0; k < C_SIZE; k++) {
//#pragma HLS UNROLL
//                sum += qkv_buff[i][k] * linear_weights_buff[k][j];
//            }
//            resultant_qkv_buff[i][j] = sum;
//        }
//    }
//}
////////////////////////////////////////////Generalized ATTENTION ///////////////////////////////////////////////////////////////////
void matrix_multiply(
    data_t A[MAX_SIZE][MAX_SIZE],
    data_t B[MAX_SIZE][MAX_SIZE],
    data_t C[MAX_SIZE][MAX_SIZE],
    int M, int N, int K,
    bool transposeB,
    bool divSqrtDModel
) {
#pragma HLS INLINE off

    // Initialize output matrix C
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
#pragma HLS PIPELINE
            C[i][j] = 0;
        }
    }

    // Perform matrix multiplication
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
#pragma HLS PIPELINE
            data_t sum = 0;
            for (int k = 0; k < N; k++) {
#pragma HLS UNROLL factor=4  // Adjust unroll factor based on your FPGA resources
                data_t a = A[i][k];
                data_t b = transposeB ? B[j][k] : B[k][j];
                sum += a * b;
            }
            if (divSqrtDModel) {
                sum /= hls::sqrt(data_t(D_MODEL));
            }
            C[i][j] = sum;
        }
    }
}

// Main function that uses the generalized matrix multiplication
void attention_mechanism(
    data_t ep_trans[SIZE][SIZE],
    data_t q_weights[SIZE][C_SIZE],
    data_t k_weights[SIZE][C_SIZE],
    data_t v_weights[SIZE][C_SIZE],
    data_t linear_weights[C_SIZE][SIZE],
    data_t resultant_qkv[SIZE][SIZE]
) {


    data_t q[SIZE][C_SIZE], k[SIZE][C_SIZE], v[SIZE][C_SIZE];
    data_t qk[SIZE][SIZE], qkv[SIZE][C_SIZE];

    // Compute Q, K, V
    matrix_multiply(ep_trans, q_weights, q, SIZE, SIZE, C_SIZE, false, false);
    matrix_multiply(ep_trans, k_weights, k, SIZE, SIZE, C_SIZE, false, false);
    matrix_multiply(ep_trans, v_weights, v, SIZE, SIZE, C_SIZE, false, false);

    // Compute QK^T
    matrix_multiply(q, k, qk, SIZE, C_SIZE, SIZE, true, true);

    // Compute (QK^T)V
    matrix_multiply(qk, v, qkv, SIZE, SIZE, C_SIZE, false, false);

    // Compute final result
    matrix_multiply(qkv, linear_weights, resultant_qkv, SIZE, C_SIZE, SIZE, false, false);
}

////////////////////////////////////////TOP  MODULE ENCODER///////////////////////////////////////////////////////////////////////////
void encoder(

//////////////////////Initializing Signals///////////////////////
		 volatile data_t *values,
		 volatile data_t *q_weights,
		 volatile data_t *k_weights ,
		 volatile data_t *v_weights,
		 volatile data_t *linear_weights,
		 volatile data_t *weights_for_linear_layer1,
		 volatile data_t *bias_layer1,
		 volatile data_t *pe,

//////////////////////INTERMEDIATE SIGNALS/////////////////////////
       volatile data_t *add_qkv_ep,
       volatile data_t *ep_trans,
       volatile data_t *resultant_qkv,

       volatile data_t *ep,

       volatile data_t *relu1_output,
       volatile data_t *linear1_output,

	   volatile data_t *normalized,

////////////////////OUTPUT SIGNAL//////////////////////////////////
	   volatile data_t *normalized_2enc


) {
	////////////////////INPUT PRAGMAS //////////////////////
#pragma HLS INTERFACE m_axi port=values offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=q_weights offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=k_weights offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi port=v_weights offset=slave bundle=gmem3
#pragma HLS INTERFACE m_axi port=linear_weights offset=slave bundle=gmem4
#pragma HLS INTERFACE m_axi port=weights_for_linear_layer1 offset=slave bundle=gmem5
#pragma HLS INTERFACE m_axi port=bias_layer1 offset=slave bundle=gmem6
#pragma HLS INTERFACE m_axi port=pe offset=slave bundle=gmem7


#pragma HLS INTERFACE m_axi port=add_qkv_ep offset=slave bundle=gmem8
#pragma HLS INTERFACE m_axi port=ep_trans offset=slave bundle=gmem9
#pragma HLS INTERFACE m_axi port=resultant_qkv offset=slave bundle=gmem10
#pragma HLS INTERFACE m_axi port=pe offset=slave bundle=gmem11
#pragma HLS INTERFACE m_axi port=ep offset=slave bundle=gmem12
#pragma HLS INTERFACE m_axi port=values offset=slave bundle=gmem13
#pragma HLS INTERFACE m_axi port=relu1_output offset=slave bundle=gmem14
#pragma HLS INTERFACE m_axi port=linear1_output offset=slave bundle=gmem15
#pragma HLS INTERFACE s_axilite port=return bundle=control


	    // Input buffers
	    data_t values_buff[SIZE][SIZE];
	    data_t q_weights_buff[SIZE][C_SIZE];
	    data_t k_weights_buff[SIZE][C_SIZE];
	    data_t v_weights_buff[SIZE][C_SIZE];
	    data_t linear_weights_buff[C_SIZE][SIZE];
	    data_t weights_for_linear_layer1_buff[SIZE][SIZE];
	    data_t bias_layer1_buff[SIZE];
	    data_t pe_buff[SIZE][SIZE];
       //Intermediate buffers
       data_t add_qkv_ep_buff[SIZE][SIZE];
       data_t ep_trans_buff[SIZE][SIZE];
       data_t resultant_qkv_buff[SIZE][SIZE];
       data_t ep_buff[SIZE][SIZE];
       data_t pe_buff[SIZE][SIZE];
       data_t values_buff[SIZE][SIZE];
       data_t relu1_output_buff[SIZE][SIZE];
       data_t linear1_output_buff[SIZE][SIZE];
       //Output buffers
       data_t normalized_2enc[SIZE][SIZE];

#pragma HLS ARRAY_PARTITION variable=values_buff complete dim=2
#pragma HLS ARRAY_PARTITION variable=q_weights_buff complete dim=2
#pragma HLS ARRAY_PARTITION variable=k_weights_buff complete dim=2
#pragma HLS ARRAY_PARTITION variable=v_weights_buff complete dim=2
#pragma HLS ARRAY_PARTITION variable=linear_weights_buff complete dim=1
#pragma HLS ARRAY_PARTITION variable=weights_for_linear_layer1_buff complete dim=2
#pragma HLS ARRAY_PARTITION variable=pe_buff complete dim=2
#pragma HLS ARRAY_PARTITION variable=q_buff complete dim=2
#pragma HLS ARRAY_PARTITION variable=k_buff complete dim=2
#pragma HLS ARRAY_PARTITION variable=v_buff complete dim=2

    #pragma HLS ARRAY_PARTITION variable=add_qkv_ep_buff complete dim=2
    #pragma HLS ARRAY_PARTITION variable=ep_trans_buff complete dim=2
    #pragma HLS ARRAY_PARTITION variable=resultant_qkv_buff complete dim=2
    #pragma HLS ARRAY_PARTITION variable=ep_buff complete dim=2
    #pragma HLS ARRAY_PARTITION variable=pe_buff complete dim=2
    #pragma HLS ARRAY_PARTITION variable=values_buff complete dim=2
    #pragma HLS ARRAY_PARTITION variable=relu1_output_buff complete dim=2
    #pragma HLS ARRAY_PARTITION variable=linear1_output_buff complete dim=2

    // Load input matrices
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
#pragma HLS PIPELINE
            ep_trans_buff[i][j] = ep_trans[i*SIZE + j];
            resultant_qkv_buff[i][j] = resultant_qkv[i*SIZE + j];
            pe_buff[i][j] = pe[i*SIZE + j];
            values_buff[i][j] = values[i*SIZE + j];
            linear1_output_buff[i][j] = linear1_output[i*SIZE + j];
        }
    }

    // Perform matrix operations
    addition(values_buff, pe_buff, ep_buff);
    transpose(ep_trans_buff, ep_buff);
    addition(ep_trans_buff, resultant_qkv_buff, add_qkv_ep_buff);
    linear_layer1(  normalized_buff,
    		  weights_for_linear_layer1 ,
    		  bias_layer1,
    		  relu1_output ,
    		  linear1_output) ;


    // Store results
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
#pragma HLS PIPELINE
            add_qkv_ep[i*SIZE + j] = add_qkv_ep_buff[i][j];
            ep[i*SIZE + j] = ep_buff[i][j];
            relu1_output[i*SIZE + j] = relu1_output_buff[i][j];
        }
    }
}


 