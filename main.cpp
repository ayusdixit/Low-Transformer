#include <iostream>
//#include "embedding.h"
using namespace std;

 //    youtube link https://www.youtube.com/watch?v=ISNdQcPhsts&t=3685s
 // github link  for scratch c++ : https://github.com/a3a256/TransformerPureC-/blob/main/include/embedding.h
 // github link of the  yoututbe channel that i am following : https://github.com/hkproj/pytorch-transformer/blob/main/model.py

 #include "preprocessing.h"
 #include "embedding.h"
 #include "multi_head.h"
 #include "activations.h"
#include "normalizing.h"
#include "encoder_linear.h"
#include "norm_add_second.h"
int main() {
    //printEncoded();   // For encoding the text 
     Matrix matrix;
     //encoder module started
    initializeMatrix(matrix);
    addMatrix(matrix) ;
    transpose(matrix) ;
    //attention
    qkv_finding(matrix);
    qk_trans(matrix);
    softmax(matrix);
    qkv_final(matrix);
    resultant_qkv(matrix);
  
    //attention block end 
      addition_block(matrix) ;
   calculate_params(matrix);
   normalize_matrix(matrix);
   //linear layer 1of encoder 
   linear_layer1(matrix) ;// consists of linear and bias layer 
   relu(matrix) ;
 // end of layer 1

   addition2_block(matrix) ;
normalize_2enc(matrix) ;
 //encode ends here
    //to print all information 
    printMatrix(matrix);
    return 0;
   
}