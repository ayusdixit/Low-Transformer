
//contains the normalizer block for decoder block with addition and normalization 
#ifndef DECODERN_H
#define DECODERN_H
const double EPSILON = 0.0001; // Small constant 
double mean[SIZE] ;
double std_dev[SIZE] ;
 
#include "embedding.h"

void d_addition_block(Matrix &m){
 for (int i = 0; i < SIZE; i++) {
     for (int j = 0; j < SIZE; j++){
    m.d_add_qkv_[i][j] = m.d_msha_norm[i][j]+m.d_resultant_qkv[i][j] ;
     }

}
}

void d_calculate_params(const Matrix& m) {
    for (int i = 0; i < SIZE; i++) {
        // Calculate mean
        double sum = 0.0;
        for (int j = 0; j < SIZE; j++) {
            sum += m.d_add_qkv_[i][j];
        }
        mean[i] = sum / SIZE;

        // Calculate standard deviation
        double variance_sum = 0.0;
        for (int j = 0; j < SIZE; j++) {
            double diff = m.d_add_qkv_[i][j] -mean[i];
            variance_sum += diff * diff;
        }
         std_dev[i] = sqrt(variance_sum / SIZE);
    }
}

void d_normalize_matrix(Matrix& m) {
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            m.d_normalized[i][j] = (m.d_add_qkv_[i][j] - mean[i]) / (std_dev[i] + EPSILON);
        }
    }
}

#endif // ATTENTION_H


