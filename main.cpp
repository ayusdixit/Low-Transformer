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
#include "decoder.h"
#include "decoder_normalize.h"
#include "decoder_linear.h"
#include "decoder_norm_add_second.h"
#include "transformer_output.h"
#include "init.h"

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
    //encoder ends here

    //decoder MHA starts 
    d_qk_trans(matrix);
    decoder_softmax(matrix);
    d_qkv_final(matrix);
    d_resultant_qkv(matrix);
    //decoder MHA ends 
    d_addition_block(matrix) ;
    d_calculate_params(matrix);
    d_normalize_matrix(matrix)  ;   //normalize block ends in decoder 
    //linear layer 1of decoder 
    d_linear_layer1(matrix) ;// consists of linear and bias layer 
    d_relu(matrix) ;
    // end of layer 1 of decoder 
    d_addition2_block(matrix) ;
    d_normalize_2enc(matrix) ;
    //to the final flatten layer of tranformer 
        final_weights_word(matrix) ;

    flattenMatrix(matrix) ;
    linear_flattenMatrix(matrix) ;
   final_softmax(matrix) ;




    //to print all information 
    printMatrix(matrix);


    return 0;

}