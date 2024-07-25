// this is initialization file for initializing the weights also 

#ifndef EMBEDDING_H
#define EMBEDDING_H

#include <iostream>
#include <cstring>
#include <cmath>
#include <iomanip>
const int SIZE = 6;
const int  D_MODEL = 6 ;
const int C_SIZE =4 ;  //used in q,k, v weights matrix init 
struct Matrix {
    char words[SIZE][10];
    int numbers[SIZE];
    double values[SIZE][SIZE];   //initial value matrix 6*6
    double pe[SIZE][SIZE];         //positional encoding matrix 6*6
    double ep[SIZE][SIZE] ;        //total input (pe + values) encoding matrix 6*6
    double ep_trans[SIZE][SIZE] ;        //transpose matrix of ep total input (pe + values) encoding matrix 6*6
    double linear_weights[C_SIZE][SIZE] ;  //linear weight matrix of 4*6
    double weights_for_linear_layer1[SIZE][SIZE] ;//considering 1 liner layer 
    double bias_layer1[1][SIZE] ; //bias matrix for linear [layer 1] 1*6

    double q_weights[SIZE][C_SIZE];   //query weights be 6*4
    double k_weights[SIZE][C_SIZE];   //key weights be 6*4
    double v_weights[SIZE][C_SIZE]; //value weights be 6*4
    double q[SIZE][C_SIZE] ;       // q matrix 
    double k[SIZE][C_SIZE] ;        // k matrix 
    double v[SIZE][C_SIZE] ;         // v matrix 
    double qk_trans[SIZE][SIZE] ;  //q*k(transpose)  it is 6*6
    double qkv_[SIZE][C_SIZE] ;   // [q*k(t)/sqrt(dk)]*V  it is 6*4 
   
    double resultant_qkv[SIZE][SIZE] ; //weights*attention filter  6*6
    double add_qkv_ep[SIZE][SIZE] ; //addition of ep + attention filter  6*6
    double normalized[SIZE][SIZE] ; //normalized*attention filter  6*6

    double linear1_output[SIZE][SIZE] ;  //output of linear layer 1 
    double relu1_output[SIZE][SIZE] ;    //output of activation function relu 1 
    double add_encoder2[SIZE][SIZE] ;   //addition of relu and normalizing  6*6
    double normalized_2enc[SIZE][SIZE] ; //normalized final output of encoder    6*6
};

void initializeMatrix(Matrix& m) {
    const char* wordList[] = {"When", "you", "play", "game", "of", "thrones"};
    int numberList[] = {5, 17, 7, 12, 15, 19};
    
    for (int i = 0; i < SIZE; i++) {
        strcpy(m.words[i], wordList[i]);
        m.numbers[i] = numberList[i];
    }

    double initValues[SIZE][SIZE] = {
        {0.79, 0.38, 0.01, 0.12, 0.88, 0.6},
        {0.6, 0.12, 0.51, 0.6, 0.41, 0.33},
        {0.96, 0.06, 0.27, 0.65, 0.79, 0.75},
        {0.64, 0.79, 0.31, 0.22, 0.62, 0.48},
        {0.97, 0.9, 0.56, 0.07, 0.5, 0.94},
        {0.2, 0.74, 0.59, 0.37, 0.7, 0.21}
    };

    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            m.values[i][j] = initValues[i][j];
        }
    }

       double q_initValues[SIZE][C_SIZE] ={
    {0.52, 0.45, 0.91, 0.69},
    {0.05, 0.85, 0.37, 0.83},
    {0.49, 0.10, 0.56, 0.61},
    {0.71, 0.64, 0.40, 0.14},
    {0.76, 0.27, 0.92, 0.67}
};

    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < C_SIZE; j++) {
            m.q_weights[i][j] = q_initValues[i][j];
        }
    }
  
    double k_initValues[SIZE][C_SIZE] = {
    {0.74, 0.57, 0.21, 0.73},
    {0.55, 0.16, 0.90, 0.17},
    {0.25, 0.74, 0.80, 0.98},
    {0.80, 0.73, 0.20, 0.31},
    {0.37, 0.96, 0.42, 0.08},
    {0.28, 0.41, 0.87, 0.86}
       };

    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < C_SIZE; j++) {
            m.k_weights[i][j] = k_initValues[i][j];
        }
    }

    double v_initValues[SIZE][C_SIZE] = {
    {0.62, 0.07, 0.70, 0.95},
    {0.20, 0.97, 0.61, 0.35},
    {0.57, 0.80, 0.61, 0.50},
    {0.67, 0.35, 0.98, 0.54},
    {0.47, 0.83, 0.34, 0.94},
    {0.60, 0.69, 0.13, 0.98}
};


    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < C_SIZE; j++) {
            m.v_weights[i][j] = v_initValues[i][j];
        }
    }

  
 double linearweights_initValues[C_SIZE][SIZE] ={
    {0.8, 0.34, 0.45, 0.54, 0.07,0.53},
    {0.85, 0.74, 0.78, 0.5 , 0.75,0.55},
    {0.53, 0.81, 0.55, 0.59, 0.49,0.14},
    {0.7, 0.6, 0.12,0.42, 0.29, 0.87} 
};

    for (int i = 0; i < C_SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            m.linear_weights[i][j] = linearweights_initValues[i][j];
        }
    }




      double layer1_linearw[SIZE][SIZE] = {
        {0.5, 0.05, 0.97, 0.22, 0.56, 0.02},
        {0.17, 0.52, 0.63, 0.48, 0.06, 0.6},
        {0.53, 0.87, 0.47, 0.1, 0.31, 0.79},
        {0.83, 0.58, 0.38, 0.09, 0.64, 0.25},
        {0.81, 0.85, 0.74, 0.35, 0.31, 0.53},
        {0.25, 0.31, 0.22, 0.77, 0.57, 0.85}
    };

    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            m.weights_for_linear_layer1[i][j] = layer1_linearw[i][j];
        }
    }

 double bias1_linearw[1][SIZE] = {      //bias OF LINEAR LAYER 1 
    {0.42, 0.18, 0.25, 0.42, 0.35, 0.45}
    
};

    for (int i = 0; i < 1; i++) {
        for (int j = 0; j < SIZE; j++) {
            m.bias_layer1[i][j] = bias1_linearw[i][j];
        }
    }


     // Calculate Position Encoding matrix
  for (int i = 0; i < SIZE; i++) {
        for (int pos = 0; pos < SIZE; pos++) {
            double angle = pos / pow(10000, (2.0 * i) / D_MODEL);
            if (i % 2 == 0) {  // Even position
                m.pe[i][pos] = sin(angle);
            } else {  // Odd position
                m.pe[i][pos] = cos(angle);
            }
        }
    }

}
void addMatrix( Matrix& m){
       for(int i = 0; i < SIZE; i++) {
        for(int j = 0; j < SIZE; j++) {
            m.ep[i][j] = m.pe[i][j] + m.values[i][j];
}
       }
}
void transpose(Matrix& m){

    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            m.ep_trans[j][i] = m.ep[i][j];
        }
    }    
}
//funciton to print helpful for debugging 
void printMatrix(const Matrix& m) {
    for (int i = 0; i < SIZE; i++) {
        std::cout << m.words[i] << "\t";
    }
    std::cout << "\n";
    for (int i = 0; i < SIZE; i++) {
        std::cout << m.numbers[i] << "\t";
    }
    std::cout << "\n\n";

    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            std::cout << m.values[i][j] << "\t";
        }
        std::cout << "\n";
    }

         std::cout << "Number of elements in values matrix: " << sizeof(m.values) / sizeof(m.values[0][0]) << std::endl;

  //positional matrix coding 
    // Print PE matrix
    std::cout << "\nPositional Encoding Matrix:\n";
      std::cout << std::setprecision(4) << std::fixed;
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            std::cout << m.pe[i][j] << "\t";
        }
        std::cout << "\n";
    }
    //input final embedding ep 
    std::cout << "\n final sum of pe+values  Encoding Matrix:\n";
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            std::cout << m.ep[i][j] << "\t";
        }
        std::cout << "\n";
    }
 std::cout << "\n --------------------------------------------------\n";
      std::cout << "\n final transposed sum of pe+values  Encoding Matrix:\n";
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            std::cout << m.ep_trans[i][j] << "\t";
        }
        std::cout << "\n";
    }
    std::cout << "\n --------------------------------------------------\n";
  std::cout << "\n query weights :\n";
 for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < C_SIZE; j++) {
            std::cout << m.q_weights[i][j] << "\t";
        }
        std::cout << "\n";
    }
std::cout << "\n --------------------------------------------------\n";
  std::cout << "\n key weights :\n";
 for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < C_SIZE; j++) {
            std::cout << m.k_weights[i][j] << "\t";
        }
        std::cout << "\n";
    }
    std::cout << "\n --------------------------------------------------\n";
      std::cout << "\n values weights :\n";
 for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < C_SIZE; j++) {
            std::cout << m.v_weights[i][j] << "\t";
        }
        std::cout << "\n";
    }



        std::cout << "\n --------------------------------------------------\n";
      std::cout << "\n final Q :\n";
 for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < C_SIZE; j++) {
            std::cout << m.q[i][j] << "\t";
        }
        std::cout << "\n";
    }
     std::cout << "\n --------------------------------------------------\n";
      std::cout << "\n final K :\n";
 for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < C_SIZE; j++) {
            std::cout << m.k[i][j] << "\t";
        }
        std::cout << "\n";
    }
     std::cout << "\n --------------------------------------------------\n";
      std::cout << "\n final V :\n";
 for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < C_SIZE; j++) {
            std::cout << m.v[i][j] << "\t";
        }
        std::cout << "\n";
    }


  std::cout << "\n --------------------------------------------------\n";
      std::cout << "\n scaled Q*K(transpose)  softmax :\n";
 for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            std::cout << m.qk_trans[i][j] << "\t";
        }
        std::cout << "\n";
    }

   std::cout << "\n --------------------------------------------------\n";
      std::cout << "\n final  softmax[Q*K(transpose)/scale]*V ]==> or Single head Attention Output   :\n";
 for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < C_SIZE; j++) {
            std::cout << m.qkv_[i][j] << "\t";
        }
        std::cout << "\n";
    }
   std::cout << "\n --------------------------------------------------\n";
      std::cout << "\n  Resultant Single head Attention Output with mult by linear weights   :\n";
 for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            std::cout << m.resultant_qkv[i][j] << "\t";
        }
        std::cout << "\n";
    }
  std::cout << "\n --------------------------------------------------\n";
      std::cout << "\n addition with ep and head attention filter   :\n";
 for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            std::cout << m.add_qkv_ep[i][j] << "\t";
        }
        std::cout << "\n";
    }

  std::cout << "\n --------------------------------------------------\n";
      std::cout << "\n  NOrmalized filter  :\n";
 for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            std::cout << m.normalized[i][j] << "\t";
        }
        std::cout << "\n";
    }
  std::cout << "\n --------------------------------------------------\n";
      std::cout << "\n  output  of linear shytt layer 1   :\n";
 for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            std::cout << m.linear1_output[i][j] << "\t";
        }
        std::cout << "\n";
    }

  std::cout << "\n --------------------------------------------------\n";
      std::cout << "\n  output  of linear  relu  layer 1   :\n";
 for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            std::cout << m.relu1_output[i][j] << "\t";
        }
        std::cout << "\n";
    }
  std::cout << "\n --------------------------------------------------\n";
      std::cout << "\n  output  of addition after  relu  layer    :\n";
 for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            std::cout << m.add_encoder2[i][j] << "\t";
        }
        std::cout << "\n";
    }
  std::cout << "\n --------------------------------------------------\n";
      std::cout << "\n  output  of encoder       :\n";
 for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            std::cout << m.normalized_2enc[i][j] << "\t";
        }
        std::cout << "\n";
    }

}
#endif // EMBEDDING_H