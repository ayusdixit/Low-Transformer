#ifndef ACTIVATION_H
#define ACTIVATION_H

#include <cmath>
#include "multi_head.h" 
 #include "embedding.h"
#include <cmath>
 
//const double EPSILON = 1e-10; // Small constant to prevent division by zero
 
 void softmax(Matrix& m) {                      //softmax activation funciton 
    for (int i = 0; i < SIZE; i++) {
        double row_sum = 0.0;
        
        // Calculate sum of exp(x) for the denominator
        for (int j = 0; j < SIZE; j++) {
            row_sum += std::exp(m.qk_trans[i][j]);
        }
        
        // Apply softmax formula to each element in the row
        for (int j = 0; j < SIZE; j++) {
            m.qk_trans[i][j] = std::exp(m.qk_trans[i][j]) / row_sum;
        }
    }
}

 void decoder_softmax(Matrix& m) {                      //softmax activation funciton 
    for (int i = 0; i < SIZE; i++) {
        double row_sum = 0.0;
        
        // Calculate sum of exp(x) for the denominator
        for (int j = 0; j < SIZE; j++) {
            row_sum += std::exp(m.d_qk_trans[i][j]);
        }
        
        // Apply softmax formula to each element in the row
        for (int j = 0; j < SIZE; j++) {
            m.d_qk_trans[i][j] = std::exp(m.d_qk_trans[i][j]) / row_sum;
        }
    }
}




void relu(Matrix& m){
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
             
                m.relu1_output[i][j] = (m.linear1_output[i][j] > 0) ? m.linear1_output[i][j] : 0;  
            }
        }
    }
 

#endif // ACTIVATION_H