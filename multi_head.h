#ifndef ATTENTION_H
#define ATTENTION_H

#include "embedding.h"

void qkv_finding(Matrix& m) {
 
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < C_SIZE; j++) {
            m.q[i][j] = 0;
            m.k[i][j] = 0;
            m.v[i][j] = 0;
        }
    }

    // Compute Q matrix
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < C_SIZE; j++) {
            for (int k = 0; k < SIZE; k++) {
                m.q[i][j] += m.ep_trans[i][k] * m.q_weights[k][j];
            }
        }
    }

    // Compute K matrix
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < C_SIZE; j++) {
            for (int k = 0; k < SIZE; k++) {
                m.k[i][j] += m.ep_trans[i][k] * m.k_weights[k][j];
            }
        }
    }

    // Compute V matrix
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < C_SIZE; j++) {
            for (int k = 0; k < SIZE; k++) {
                m.v[i][j] += m.ep_trans[i][k] * m.v_weights[k][j];
            }
        }
    }


}


void qk_trans(Matrix& m) {
    // Initialize qk_result matrix important
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            m.qk_trans[i][j] = 0;
        }
    }

    // Perform matrix multiplication: q * k^T
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            for (int k = 0; k < C_SIZE; k++) {
                m.qk_trans[i][j] += (m.q[i][k] * m.k[j][k])/sqrt(D_MODEL);  // Note: k[j][k] instead of k[k][j] for transpose
            }
        }
    }

}

void qkv_final(Matrix& m){
 // Initialize qk_result matrix important
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < C_SIZE; j++) {
            m.qkv_[i][j] = 0;
        }
 
    }
  // Perform matrix multiplication:  qk*V
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < C_SIZE; j++) {
            for (int k = 0; k < SIZE; k++) {
                m.qkv_[i][j] += (m.qk_trans[i][k] * m.v[j][k]);  // Note: k[j][k] instead of k[k][j] for transpose
            }
        }
    }


}
void resultant_qkv(Matrix& m){
  // Initialize qk_result matrix important
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            m.resultant_qkv[i][j] = 0;
        }
 
    }  


  // Perform matrix multiplication:  qk*V
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            for (int k = 0; k < C_SIZE; k++) {
                m.resultant_qkv[i][j] += (m.qkv_[i][k] * m.linear_weights[j][k]);  // Note: k[j][k] instead of k[k][j] for transpose
            }
        }
    }


}

#endif // ATTENTION_H
