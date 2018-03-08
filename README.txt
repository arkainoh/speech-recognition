@title:  README.txt
@author: Inho, Kim
@e-mail: arkainoh@gmail.com

# Files
 locate these files in the same directory

          "tst/": test dataset
   "hmm_utils.h": header file for hmm_utils.c
   "hmm_utils.c": implementation of core functions (e.g. viterbi, get_transmat, get_emissmat, etc.) 
        "main.c": implementation of main function
"dictionary.txt": set of vocabularies and their monophones
   "unigram.txt": unigram probabilities of each word
    "bigram.txt": bigram probabilities of each word pair
 "reference.txt": labels of "tst/" test dataset
  "HResults.exe": program to get test accuracy
       "hmm.txt": hmm parameters
         "hmm.h": implementation of c structure based on "hmm.txt"
       

# Compilation
 gcc hmm_utils.c main.c -o HRecognize

# Execution
 (1) ./HRecognize.exe: recognize function is called with a path of the test dataset as an argument ("tst/")
                       output filename is set to "recognized.txt"

 (2)      by yourself: recognize(<indir>, <outfile>) function conducts the whole process of the test
                       <indir>:   path of the test dataset
                       <outfile>: path of the result file
