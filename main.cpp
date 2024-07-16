#include <iostream>
#include "embedding.h"
using namespace std;

 //    youtube link https://www.youtube.com/watch?v=ISNdQcPhsts&t=3685s
 // github link  for scratch c++ : https://github.com/a3a256/TransformerPureC-/blob/main/include/embedding.h
 // github link of the  yoututbe channel that i am following : https://github.com/hkproj/pytorch-transformer/blob/main/model.py

 
//Main function
int main() {
    const int d_model = 512;
    const int vocab_size = 10;

    // Create an instance of InputEmbeddings
    InputEmbeddings embeddings(d_model, vocab_size);
    embeddings.forward();
    // // Example: Print a sample embedding value
    cout << "Sample embedding[0][0]:" << embeddings.embedding[0][0] << endl;

    return 0;
}