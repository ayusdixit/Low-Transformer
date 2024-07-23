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

    double q_weights[SIZE][C_SIZE];   //query weights be 6*4
    double k_weights[SIZE][C_SIZE];   //key weights be 6*4
    double v_weights[SIZE][C_SIZE]; //value weights be 6*4
    double q[SIZE][C_SIZE] ;       // q matrix 
    double k[SIZE][C_SIZE] ;        // k matrix 
    double v[SIZE][C_SIZE] ;         // v matrix 
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
}
#endif // EMBEDDING_H