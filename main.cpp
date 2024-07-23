#include <iostream>
//#include "embedding.h"
using namespace std;

 //    youtube link https://www.youtube.com/watch?v=ISNdQcPhsts&t=3685s
 // github link  for scratch c++ : https://github.com/a3a256/TransformerPureC-/blob/main/include/embedding.h
 // github link of the  yoututbe channel that i am following : https://github.com/hkproj/pytorch-transformer/blob/main/model.py

 #include "preprocessing.h"
 #include "embedding.h"
 #include "multi_head.h"
 

int main() {
    //printEncoded();   // For encoding the text 
     Matrix matrix;
    initializeMatrix(matrix);
    addMatrix(matrix) ;
    transpose(matrix) ;
    qkv_finding(matrix);
    printMatrix(matrix);
    return 0;
   
}