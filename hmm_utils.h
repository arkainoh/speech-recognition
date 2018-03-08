#ifndef HMM_UTILS
#define HMM_UTILS

#include <stdlib.h>
#include <stdio.h>

#define L_BUFFER 128
#define MAX_N_WORD 13
#define MAX_L_WORD 6
#define MAX_N_PHONE 6
#define MAX_L_PHONE 3
#define L_UTTERANCE 7

typedef struct {
	int rows, cols;
	double** elements;
} matrix;

typedef struct {
	char str[MAX_L_WORD];
	int index;
	int n_phone;
	char phones[MAX_N_PHONE][MAX_L_PHONE];
	int hmm_indices[MAX_N_PHONE];
} word;

typedef struct {
	int size;
	int n_word;
	word words[MAX_N_WORD];
} dictionary;

typedef struct {
	int size;
	double prob[MAX_N_WORD];
} unigram;

typedef struct {
	int size;
	double prob[MAX_N_WORD][MAX_N_WORD];
} bigram;

void init_unigram(unigram* ug);
void init_bigram(bigram* bg);
void recognize(char* indir, char* outfile);
dictionary get_dictionary(char* filename);
int word_index(dictionary* dic, char* w);
unigram get_unigram(dictionary* dic, char* filename);
bigram get_bigram(dictionary* dic, char* filename);
void test_inputs(char* dirname, dictionary* dic, int labels[MAX_N_WORD], matrix* a, FILE* f);
matrix read_input(char* filename);
void remove_ext(char (*str)[]);
matrix new_matrix(int m, int n);
void free_matrix(matrix* mat);
double emission(int hmm_index, int state, matrix* x, int t);
int* viterbi(matrix* a, matrix* b);
matrix get_transmat(dictionary* dic, unigram* ug, bigram* bg, int labels[MAX_N_WORD]);
matrix get_emissmat(dictionary* dic, matrix* x, int labels[MAX_N_WORD]);
int* get_labels(dictionary* dic);
void print_results(FILE* f, dictionary* dic, int* state_seq, int length, int labels[MAX_N_WORD]);
void print_matrix(matrix* mat);
#endif