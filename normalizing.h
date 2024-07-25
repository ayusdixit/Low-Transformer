#ifndef  NORMALIZING_H
#define NORMALIZING_H 
#include "embedding.h"
#include "multi_head.h"

const double EPSILON = 0.0001; // Small constant 
double mean[SIZE] ;
double std_dev[SIZE] ;

void addition_block(Matrix &m){
 for (int i = 0; i < SIZE; i++) {
     for (int j = 0; j < SIZE; j++){
    m.add_qkv_ep[i][j] = m.ep_trans[i][j]+m.resultant_qkv[i][j] ;
     }

}
}

void calculate_params(const Matrix& m) {
    for (int i = 0; i < SIZE; i++) {
        // Calculate mean
        double sum = 0.0;
        for (int j = 0; j < SIZE; j++) {
            sum += m.resultant_qkv[i][j];
        }
        mean[i] = sum / SIZE;

        // Calculate standard deviation
        double variance_sum = 0.0;
        for (int j = 0; j < SIZE; j++) {
            double diff = m.resultant_qkv[i][j] -mean[i];
            variance_sum += diff * diff;
        }
         std_dev[i] = sqrt(variance_sum / SIZE);
    }
}

void normalize_matrix(Matrix& m) {
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            m.normalized[i][j] = (m.add_qkv_ep[i][j] - mean[i]) / (std_dev[i] + EPSILON);
        }
    }
}











#endif 