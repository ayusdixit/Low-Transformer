#ifndef ENC_NORM_H
#define ENC_NORM_H
  
 #include "preprocessing.h"
 #include "embedding.h"
 #include "multi_head.h"
 #include "activations.h"
#include "normalizing.h"
#include "encoder_linear.h"
#include "norm_add_second.h"
double mean1[SIZE] ;
double std_dev1[SIZE] ;
 
void addition2_block(Matrix& m){
 for (int i = 0; i < SIZE; i++) {
     for (int j = 0; j < SIZE; j++){
    m.add_encoder2[i][j] = m.relu1_output[i][j]+m.normalized[i][j] ;
     }

}
} 
void normalize_2enc(Matrix& m) {
    for (int i = 0; i < SIZE; i++) {
        // Calculate mean
        double sum = 0.0;
        for (int j = 0; j < SIZE; j++) {
            sum += m.add_encoder2[i][j];
        }
        mean[i] = sum / SIZE;

        // Calculate standard deviation
        double variance_sum = 0.0;
        for (int j = 0; j < SIZE; j++) {
            double diff = m.add_encoder2[i][j] -mean[i];
            variance_sum += diff * diff;
        }
         std_dev1[i] = sqrt(variance_sum / SIZE);
  
 
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            m.normalized_2enc[i][j] = (m.add_encoder2[i][j] - mean[i]) / (std_dev[i] + EPSILON);
        }
    }
}
}

 
#endif  