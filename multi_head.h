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

#endif // ATTENTION_H
