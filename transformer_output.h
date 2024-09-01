#ifndef TP_H
#define TP_H

  #include "decoder.h"
#include "decoder_normalize.h"
#include "decoder_linear.h"
#include "decoder_norm_add_second.h"
#include "transformer_output.h"
 #include "preprocessing.h"
 #include "embedding.h"
 #include "multi_head.h"
 #include "activations.h"
#include "normalizing.h"
#include "encoder_linear.h"
#include "norm_add_second.h"

void flattenMatrix(Matrix& m) {
    //initialize to zero 
     for (int i = 0; i < SIZE*SIZE; i++) {
       
            m.flattened[i] = 0;
    }  

    int index = 0;  // Index for the flatArray

    // Iterate over each row of the matrix
    for (int i = 0; i < SIZE; ++i) {
        // Iterate over each column of the matrix
        for (int j = 0; j < SIZE; ++j) {
            // Copy the matrix element into the flatArray
            m.flattened[index++] =m.d_normalized_2enc[i][j];
        }
    }
}
void linear_flattenMatrix(Matrix& m) {
 
 // Initialize linear_logits array
    for (int j = 0; j < 23; ++j) {
        m.linear_logits[j] = 0.0f;
    }

    // Perform matrix multiplication: flattened * weights_output_final_linear
    for (int j = 0; j < 23; ++j) {
        float sum = 0.0f;
        for (int k = 0; k < 36; ++k) {
            sum += m.flattened[k] * m.weights_output_final_linear[k][j];
        }
        m.linear_logits[j] = sum;
    }
}





#endif