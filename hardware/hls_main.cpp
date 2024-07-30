#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ap_fixed.h"
#include "hls_math.h"
#define SIZE  36
#define C_SIZE 4
#define D_MODEL 6
const double EPSILON = 0.0001; // Small constant
double mean[SIZE] ;
double std_dev[SIZE] ;

typedef ap_fixed<8, 2> data_t;


////////////////////////////////////////NORMALISATION LAYER///////////////////////////////////////////////////////////////////////////
void normalize_matrix(

    data_t resultant_qkv_buff[SIZE][SIZE],
    data_t normalized_buff[SIZE][SIZE]
) {
    data_t mean[SIZE];
    data_t std_dev[SIZE];

    // Convert EPSILON to data_t
    const data_t epsilon = static_cast<data_t>(EPSILON);

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
            normalized_buff[i][j] = (resultant_qkv_buff[i][j] - mean[i]) / (std_dev[i] + epsilon);
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
void qkv_finding(data_t ep_trans[SIZE][SIZE], data_t q_weights[SIZE][C_SIZE], data_t k_weights[SIZE][C_SIZE], data_t v_weights[SIZE][C_SIZE],
        data_t q[SIZE][C_SIZE], data_t k[SIZE][C_SIZE], data_t v[SIZE][C_SIZE]) {
    // Initialize Q, K, V matrices
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < C_SIZE; j++) {
            #pragma HLS PIPELINE
            q[i][j] = 0;
            k[i][j] = 0;
            v[i][j] = 0;
        }
    }
    // Compute Q matrix
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < C_SIZE; j++) {
            #pragma HLS PIPELINE
            for (int m = 0; m < SIZE; m++) {
                q[i][j] += ep_trans[i][m] * q_weights[m][j];
            }
        }
    }
    // Compute K matrix
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < C_SIZE; j++) {
            #pragma HLS PIPELINE
            for (int m = 0; m < SIZE; m++) {
                k[i][j] += ep_trans[i][m] * k_weights[m][j];
            }
        }
    }
    // Compute V matrix
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < C_SIZE; j++) {
            #pragma HLS PIPELINE
            for (int m = 0; m < SIZE; m++) {
                v[i][j] += ep_trans[i][m] * v_weights[m][j];
            }
        }
    }
}

void qk_transs(data_t q[SIZE][C_SIZE], data_t k[SIZE][C_SIZE], data_t qk_trans[SIZE][SIZE]) {
    // Initialize qk_trans
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            #pragma HLS PIPELINE
            qk_trans[i][j] = 0;
        }
    }

    // Perform matrix multiplication: q * k^T
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            #pragma HLS PIPELINE
            for (int m = 0; m < C_SIZE; m++) {
                qk_trans[i][j] += q[i][m] * k[j][m] / hls::sqrt(D_MODEL);  // Note: k[j][m] for transpose
                softmax(qk_trans);
            }
        }
    }
}

void qkv_final(data_t qkv_[SIZE][C_SIZE], data_t qk_trans[SIZE][SIZE], data_t v[SIZE][C_SIZE]) {

    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < C_SIZE; j++) {
#pragma HLS PIPELINE
        	qkv_[i][j] = 0;
        }
    }

    // Perform matrix multiplication: qk*V
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < C_SIZE; j++) {
#pragma HLS PIPELINE
            for (int k = 0; k < SIZE; k++) {
            	qkv_[i][j] += qk_trans[i][k] * v[k][j];
            }
        }
    }
}




void resultant_qkvv(data_t qkv_[SIZE][C_SIZE], data_t linear_weights[C_SIZE][SIZE], data_t resultant_qkv[SIZE][SIZE]) {
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            #pragma HLS PIPELINE
            resultant_qkv[i][j] = 0;
        }
    }
    // Perform matrix multiplication: qk*V
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            #pragma HLS PIPELINE
            data_t sum = 0;
            for (int k = 0; k < C_SIZE; k++) {
                #pragma HLS UNROLL
                sum += qkv_[i][k] * linear_weights[k][j];
            }
            resultant_qkv[i][j] = sum;
        }
    }
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
		 volatile data_t *ep_trans,

		 volatile data_t *q ,
		 volatile data_t *k ,
		 volatile data_t *v ,
         volatile data_t *qk_trans,
		 volatile data_t *qkv_,





        volatile data_t *add_encoder2,
		 volatile data_t *add_qkv_ep,

       volatile data_t *resultant_qkv,

       volatile data_t *ep,

       volatile data_t *relu1_output,
       volatile data_t *linear1_output,

   volatile data_t *normalized ,

//
//////////////////////OUTPUT SIGNAL//////////////////////////////////
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
	    data_t q_buff[SIZE][C_SIZE];
	    data_t k_buff[SIZE][C_SIZE];
	    data_t v_buff[SIZE][C_SIZE];

	    data_t qk_trans_buff[SIZE][SIZE];
	    data_t qkv_buff[SIZE][C_SIZE];


	     data_t resultant_qkv_buff[SIZE][SIZE];

       data_t add_qkv_ep_buff[SIZE][SIZE];
       data_t ep_trans_buff[SIZE][SIZE];

       data_t ep_buff[SIZE][SIZE];


       data_t relu1_output_buff[SIZE][SIZE];
       data_t linear1_output_buff[SIZE][SIZE];


       data_t add_encoder2_buff[SIZE][SIZE];
       data_t normalized_buff[SIZE][SIZE];
       data_t normalized_2enc_buff[SIZE][SIZE];

//       //Output buffers
//       data_t normalized_2enc[SIZE][SIZE];

#pragma HLS ARRAY_PARTITION variable=values_buff complete dim=2
#pragma HLS ARRAY_PARTITION variable=q_weights_buff complete dim=2
#pragma HLS ARRAY_PARTITION variable=k_weights_buff complete dim=2
#pragma HLS ARRAY_PARTITION variable=v_weights_buff complete dim=2
#pragma HLS ARRAY_PARTITION variable=linear_weights_buff complete dim=1
#pragma HLS ARRAY_PARTITION variable=weights_for_linear_layer1_buff complete dim=2
#pragma HLS ARRAY_PARTITION variable=pe_buff complete dim=2


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
        	   values_buff[i][j] = values[i*SIZE + j];

//            resultant_qkv_buff[i][j] = resultant_qkv[i*SIZE + j];
            pe_buff[i][j] = pe[i*SIZE + j];
//
            linear1_output_buff[i][j] = linear1_output[i*SIZE + j];
        }
    }

    // Load weight matrices
      for (int i = 0; i < SIZE; i++) {
          for (int j = 0; j < C_SIZE; j++) {
              #pragma HLS PIPELINE
              q_weights_buff[i][j] = q_weights[i*C_SIZE + j];
              k_weights_buff[i][j] = k_weights[i*C_SIZE + j];
              v_weights_buff[i][j] = v_weights[i*C_SIZE + j];
          }
      }


    // Perform matrix operations
    addition(values_buff, pe_buff, ep_buff);

//    addition(ep_trans_buff, resultant_qkv_buff, add_qkv_ep_buff);
//    linear_layer1(  normalized_buff,
//    		  weights_for_linear_layer1 ,
//    		  bias_layer1,
//    		  relu1_output ,
//    		  linear1_output) ;


    // Store results
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
#pragma HLS PIPELINE
            ep[i*SIZE + j] = ep_buff[i][j];
//            add_qkv_ep[i*SIZE + j] = add_qkv_ep_buff[i][j];
//
//            relu1_output[i*SIZE + j] = relu1_output_buff[i][j];
        }
    }

       transpose(ep_trans_buff, ep_buff);
                                                             // Store ep_trans_buff to ep_trans
         for (int i = 0; i < SIZE; i++) {
             for (int j = 0; j < SIZE; j++) {
                 #pragma HLS PIPELINE
                 ep_trans[i*SIZE + j] = ep_trans_buff[i][j];   //this is the output
             }
         }

         qkv_finding(ep_trans_buff, q_weights_buff, k_weights_buff, v_weights_buff, q_buff, k_buff, v_buff);
         for (int i = 0; i < SIZE; i++) {
                for (int j = 0; j < C_SIZE; j++) {
                    #pragma HLS PIPELINE
                    q[i*C_SIZE + j] = q_buff[i][j];
                    k[(SIZE+i)*C_SIZE + j] = k_buff[i][j];
                    v[(2*SIZE+i)*C_SIZE + j] = v_buff[i][j];
                }
            }


         qk_transs( q_buff,k_buff,qk_trans_buff);

           for (int i = 0; i < SIZE; i++) {
               for (int j = 0; j < SIZE; j++) {
                   #pragma HLS PIPELINE
                   qk_trans[i*SIZE + j] = qk_trans_buff[i][j];
               }
           }

           qkv_final( qkv_buff ,   qk_trans_buff,   v_buff );
           for (int i = 0; i < SIZE; i++) {
                   for (int j = 0; j < C_SIZE; j++) {
                       #pragma HLS PIPELINE
                       qkv_[i*C_SIZE + j] = qkv_buff[i][j];
                   }
               }


           resultant_qkvv(qkv_buff, linear_weights_buff , resultant_qkv_buff );
           // Write resultant_qkv_buff to output
             for (int i = 0; i < SIZE; i++) {
                 for (int j = 0; j < SIZE; j++) {
                     #pragma HLS PIPELINE
                	 resultant_qkv[i*SIZE + j] = resultant_qkv_buff[i][j];
                 }
             }

             addition(ep_trans_buff, resultant_qkv_buff, add_qkv_ep_buff);

                       for (int i = 0; i < SIZE; i++) {
                           for (int j = 0; j < SIZE; j++) {
                               #pragma HLS PIPELINE
                        	   add_qkv_ep[i*SIZE + j] = add_qkv_ep_buff[i][j];
                           }
                       }


           normalize_matrix(resultant_qkv_buff ,normalized_buff  )  ;
           for (int i = 0; i < SIZE; i++) {
                                for (int j = 0; j < SIZE; j++) {
                                    #pragma HLS PIPELINE
                                	normalized[i*SIZE + j] = normalized_buff[i][j];
                                }
                            }


           linear_layer1(normalized_buff, weights_for_linear_layer1_buff, bias_layer1_buff,  relu1_output_buff,linear1_output_buff)  ;
           for (int i = 0; i < SIZE; i++) {
                                         for (int j = 0; j < SIZE; j++) {
                                             #pragma HLS PIPELINE
                                        	 linear1_output[i*SIZE + j] = linear1_output_buff[i][j];
                                         }
                                     }

            addition( normalized_buff, relu1_output_buff, add_encoder2_buff);
             for (int i = 0; i < SIZE; i++) {
                                  for (int j = 0; j < SIZE; j++) {
                                      #pragma HLS PIPELINE
                                	  add_encoder2[i*SIZE + j] = add_encoder2_buff[i][j];
                                  }
                              }

             normalize_matrix(add_encoder2_buff ,normalized_2enc_buff  )  ;
             for (int i = 0; i < SIZE; i++) {
                                        for (int j = 0; j < SIZE; j++) {
                                            #pragma HLS PIPELINE
                                        	normalized_2enc[i*SIZE + j] = normalized_2enc_buff[i][j];
                                        }
                                    }

       }





