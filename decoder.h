
//  contains MHA block for decoder block

#ifndef DECODER_H
#define DECODER_H

 
#include "embedding.h"



void d_qk_trans(Matrix& m) {
    // Initialize d_qk_result matrix important
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            m.d_qk_trans[i][j] = 0;
        }
    }

    // Perform matrix multiplication decoder : q * k^T
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            for (int k = 0; k < SIZE; k++) {
                m.d_qk_trans[i][j] += (m.normalized_2enc[i][k] * m.normalized_2enc[j][k])/sqrt(D_MODEL);  // transposed multiplication 
            }
        }
    }

}
void d_qkv_final(Matrix& m)  {
 // Initialize qk_result matrix important
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < C_SIZE; j++) {
            m.d_qkv_[i][j] = 0;
        }
 
    }
  // Perform matrix multiplication:  qk*V
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            for (int k = 0; k < SIZE; k++) {
                m.d_qkv_[i][j] += (m.d_qk_trans[i][k] * m.d_v[k][j]);  // Note: k[j][k] instead of k[k][j] for transpose
            }
        }
    }


}
void d_resultant_qkv(Matrix& m){
  // Initialize qk_result matrix important
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            m.d_resultant_qkv[i][j] = 0;
        }
 
    }  


  // Perform matrix multiplication:  qk*V
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            for (int k = 0; k < C_SIZE; k++) {
                m.d_resultant_qkv[i][j] += (m.qkv_[i][k] * m.d_linear_weights[j][k]);  // Note: k[j][k] instead of k[k][j] for transpose
            }
        }
    }


}





#endif // EMBEDDING_H