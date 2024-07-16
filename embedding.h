 
#include <iostream>
#include <cstdlib>  
#include <ctime>    
#include <cmath>
using namespace std;

//  InputEmbeddings class
class InputEmbeddings {
public:
    float embedding[512][512]; // Assuming d_model = 512 and vocab_size = 512  , have to change this during changing the vocab_size()

    // Constructor
    InputEmbeddings(int d_model, int vocab_size) {
        srand(time(nullptr)); // Initialize random seed

        // Initialize embeddings with random values
        for (int i = 0; i < vocab_size; ++i) {
            for (int j = 0; j < d_model; ++j) {
                embedding[i][j] = random_float(); // Assuming random_float() returns a random float value
            }
        }
    }
   void forward() {
        // Scale the embeddings by sqrt(d_model)
        float scale = sqrt(512); // Assuming d_model = 512
        for (int i = 0; i < 512; ++i) {
            for (int j = 0; j < 512; ++j) {
                embedding[i][j] *= scale;
            }
        }
    }

private:
 
    float random_float() {
        return static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }
};

class Positional_encoding {

Positional_encoding( int d_model , int seq_len , float drop_out){
//to create a matrix of size d_model * seq_len
int pe[d_model][seq_len]  ;
 for (int i = 0; i < d_model; ++i) {
            for (int j = 0; j < seq_len; ++j) {
       pe[i][j] = 0 ;
}} 
//to create a vector of lenght of seq_len*1
 float position[seq_len][1];
    for (int i = 0; i < seq_len; ++i) {
        position[i][0] = static_cast<float>(i);
    }
  float div_term[d_model / 2];       //i  think there is some missing part 
    for (int i = 0; i < d_model / 2; ++i) {
        div_term[i] = std::exp(static_cast<float>(i * 2) * (-std::log(10000.0) / d_model));
    }

}};
 