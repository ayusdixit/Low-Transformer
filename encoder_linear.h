#ifndef LINEAR1_H
#define LINEAR1_H
 #include <iostream>
 #include "preprocessing.h"
 #include "embedding.h"
 #include "multi_head.h"
 #include "activations.h"
#include "normalizing.h"
 
void linear_layer1(Matrix& m) {
     double result[6][6];
    // Initialize linear1_output to 0
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            m.linear1_output[i][j] = 0;
        }
    }
      std::cout << "\n --------------------------------------------------\n";
      std::cout << "\n  NOrmalized filter part2  :\n";
 for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            std::cout << m.normalized[i][j] << "\t";
        }
        std::cout << "\n";
    }
   std::cout << "\n --------------------------------------------------\n";
      std::cout << "\n  weights to check    part2  :\n";
 for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            std::cout << m.weights_for_linear_layer1[i][j] << "\t";
        }
        std::cout << "\n";
    }

 for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            for(int k=0;k< SIZE ; k++){
                m.linear1_output[i][j] += (m.normalized[i][k] * m.weights_for_linear_layer1[k][j]) ;
            }
        }}

 
 
    // Add bias to each element
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            m.linear1_output[i][j] += m.bias_layer1[0][j];
        }
    }
   std::cout << "\n --------------------------------------------------\n";
      std::cout << "\n  linear ;auer output  filter part2  :\n";
 for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            std::cout << m.linear1_output[i][j] << "\t";
        }
        std::cout << "\n";
    }
}


#endif  