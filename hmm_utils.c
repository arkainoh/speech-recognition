#include "hmm_utils.h"
#include "hmm.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>

#define MAX_N_HMM sizeof(hmm) / sizeof(hmmType)

/*
 * viterbi
 * @input: preconstructed transition probability a, emission probability b (j x t matrices)
 * @output: state sequence q (1 x t vector)
 */
int* viterbi(matrix* a, matrix* b) {
	double** delta; // j x t
	int** psi; // j x t
	int* q; // 1 x t
	
	delta = (double**)malloc(a->rows * sizeof(double*));
	psi = (int**)malloc(a->rows * sizeof(int*));
	for(int j = 0; j < a->rows; j++) {
		delta[j] = (double*)malloc(b->cols * sizeof(double));
		psi[j] = (int*)malloc(b->cols * sizeof(int));
	}
	
	q = (int*)malloc(b->cols * sizeof(int));
	
	for(int j = 0; j < a->rows; j++) {
		for(int t = 0; t < b->cols; t++) {
			delta[j][t] = 0;
			psi[j][t] = 0;
		}
	}
	
	for(int t = 0; t < b->cols; t++) {
		q[t] = 0;
	}
	
	// init
	for(int j = 0; j < a->rows; j++) {
		delta[j][0] = a->elements[0][j] + b->elements[j][0];
	}
	
	// get delta and psi with recursion
	for(int t = 1; t < b->cols; t++) {
		for(int j = 0; j < a->rows; j++) {
			double max_delta = delta[0][t - 1] + a->elements[0][j] + b->elements[j][t];
			
			for(int i = 1; i < a->rows; i++) {
				double tmp = delta[i][t - 1] + a->elements[i][j] + b->elements[j][t];
				
				if(max_delta < tmp) {
					max_delta = tmp;
					psi[j][t] = i;
				}
			}
			delta[j][t] = max_delta;
		}
	}
	
	double max_delta = delta[0][b->cols - 2];
	q[b->cols - 2] = 0;
	
	
	for(int j = 1; j < a->rows; j++) {
		double tmp = delta[j][b->cols - 2];
		if(max_delta < tmp) {
			max_delta = tmp;
			q[b->cols - 2] = j;
		}
	}

	// get state sequence
	for(int t = b->cols - 3; t >= 0; t--) {
		q[t] = psi[q[t + 1]][t + 1];
	}
	
	for(int i = 0; i < a->rows; i++) {
		free(delta[i]);
		free(psi[i]);
	}
	
	free(delta);
	free(psi);
	return q;
}

void print_results(FILE* f, dictionary* dic, int* state_seq, int length, int labels[MAX_N_WORD]) {
	int new_word = 1;
	int t = 0;
	
	int initial_state = state_seq[0];
	if(initial_state != 1) { // remove initial noise
		 while(state_seq[t] == initial_state) t++;
	}
	
	for(; t < length; t++) {
		
		for(int w = 0; w < dic->size; w++) {
			
			if(state_seq[t] == labels[w] + 1 && state_seq[t] != labels[MAX_N_WORD - 1] + 1) {
				if(new_word) {
					if(f) fprintf(f, "%s\n", dic->words[w + 1].str);
					printf("%s\n", dic->words[w + 1].str);
					new_word = 0;
				}
			}
			
			if(state_seq[t] == labels[w] || state_seq[t] == labels[w] - 1) {
				new_word = 1;
			}
		}
	}
}

double emission(int hmm_index, int state, matrix* x, int t) {
	double l[N_PDF];
	for(int g = 0; g < N_PDF; g++) {
		l[g] = log(hmm[hmm_index].state[state].pdf[g].weight) - (N_DIMENSION / 2) * log(2 * M_PI);
		for(int i = 0; i < N_DIMENSION; i++) {
			l[g] = l[g] - log(sqrt(hmm[hmm_index].state[state].pdf[g].var[i])) - 0.5 * pow(x->elements[t][i] - hmm[hmm_index].state[state].pdf[g].mean[i], 2) / (hmm[hmm_index].state[state].pdf[g].var[i]);
		}
	}
	
	double sum_exp = 0;

	double max_val = l[0];
	for(int g = 1; g < N_PDF; g++) {
		if(max_val < l[g]) max_val = l[g];
	}
	
	for(int g = 0; g < N_PDF; g++) {
		sum_exp += exp(l[g] - max_val);
	}
	
	return (max_val + log(sum_exp));
}

int* get_labels(dictionary* dic) {
	int start_index, start_indices[MAX_N_WORD];
	int* end_indices = (int*)malloc(sizeof(int) * MAX_N_WORD);
	
	// get the number of states
	start_index = 1;
	for(int w = 0; w < dic->size; w++) {
		int count = 0;
		start_indices[w] = start_index;
		
		for(int p = 0; p < dic->words[w].n_phone; p++) {
			count += hmm[dic->words[w].hmm_indices[p]].n_state;
		}
		
		start_index += count;
		end_indices[w] = start_index - 1;
	}
	
	return end_indices;
}

matrix get_transmat(dictionary* dic, unigram* ug, bigram* bg, int labels[MAX_N_WORD]) {
	matrix a;
	
	int n_state, start_index, start_indices[MAX_N_WORD];
	
	n_state = labels[MAX_N_WORD - 1] + 2;
	start_indices[0] = 1;
	
	for(int w = 1; w < dic->size; w++) {
		start_indices[w] = labels[w - 1] + 1;
	}
	
	a = new_matrix(n_state, n_state); // make a matrix with padding
	
	for(int i = 0; i < dic->size; i++) {
		// in-link (1st row)
		a.elements[0][start_indices[i]] += log(hmm[dic->words[i].hmm_indices[0]].tp[0][1]) + log(ug->prob[dic->words[i].index]);
		
		// out-link (1st column)
		a.elements[labels[i] - 1][1] += log(hmm[dic->words[i].hmm_indices[dic->words[i].n_phone - 1]].tp[hmm[dic->words[i].hmm_indices[dic->words[i].n_phone - 1]].n_state - 1][hmm[dic->words[i].hmm_indices[dic->words[i].n_phone - 1]].n_state + 1]);
		a.elements[labels[i]][1] += log(hmm[dic->words[i].hmm_indices[dic->words[i].n_phone - 1]].tp[hmm[dic->words[i].hmm_indices[dic->words[i].n_phone - 1]].n_state][hmm[dic->words[i].hmm_indices[dic->words[i].n_phone - 1]].n_state + 1]);
		
		for(int j = 0; j < dic->size; j++) {
			
			// in-link
			a.elements[labels[i]][start_indices[j]] += log(hmm[dic->words[j].hmm_indices[0]].tp[0][1]) + log(bg->prob[dic->words[i].index][dic->words[j].index]); //remove unigram
			// if the former word ends with sp, the latter word should consider the case when sp has not occurred
			if(i > 0) a.elements[labels[i] - 1][start_indices[j]] += log(hmm[dic->words[j].hmm_indices[0]].tp[0][1]) + log(bg->prob[dic->words[i].index][dic->words[j].index]); // remove unigram
			
			// out-link
			a.elements[labels[i] - 1][labels[j] + 1] += log(hmm[dic->words[i].hmm_indices[dic->words[i].n_phone - 1]].tp[hmm[dic->words[i].hmm_indices[dic->words[i].n_phone - 1]].n_state - 1][hmm[dic->words[i].hmm_indices[dic->words[i].n_phone - 1]].n_state + 1]);
			a.elements[labels[i]][labels[j] + 1] += log(hmm[dic->words[i].hmm_indices[dic->words[i].n_phone - 1]].tp[hmm[dic->words[i].hmm_indices[dic->words[i].n_phone - 1]].n_state][hmm[dic->words[i].hmm_indices[dic->words[i].n_phone - 1]].n_state + 1]);
		}
	}
	
	// <s> can transit from state 3 to 1 by itself
	a.elements[3][1] = 0;
	
	// copy the rest
	for(int w = 0; w < dic->size; w++) {
		start_index = start_indices[w];
		for(int p = 0; p < dic->words[w].n_phone; p++) {
			
			// if p is not the first phone
			if(p != 0)
				a.elements[start_index - 1][start_index] += log(hmm[dic->words[w].hmm_indices[p]].tp[0][1]);
				
			// if p is not the last phone
			if(p != dic->words[w].n_phone - 1)
				a.elements[start_index + hmm[dic->words[w].hmm_indices[p]].n_state - 1][start_index + hmm[dic->words[w].hmm_indices[p]].n_state] += log(hmm[dic->words[w].hmm_indices[p]].tp[hmm[dic->words[w].hmm_indices[p]].n_state][hmm[dic->words[w].hmm_indices[p]].n_state + 1]);
			
			for(int i = 0; i < hmm[dic->words[w].hmm_indices[p]].n_state; i++) {
				for(int j = 0; j < hmm[dic->words[w].hmm_indices[p]].n_state; j++) {
					if(i == 2 && j == 0) { // if a word is not <s>, it can't transit from state 3 to 1 by itself
						if(w == 0) {
							a.elements[start_index + i][start_index + j] += log(hmm[dic->words[w].hmm_indices[p]].tp[i + 1][j + 1]);
						}
					} else {
						a.elements[start_index + i][start_index + j] += log(hmm[dic->words[w].hmm_indices[p]].tp[i + 1][j + 1]);
					}
				}
			}
			
			start_index += hmm[dic->words[w].hmm_indices[p]].n_state;
		}
	}
	
	// convert 0 to log(0)
	for(int i = 0; i < a.rows; i++) {
		for(int j = 0; j < a.cols; j++) {
			if(a.elements[i][j] == 0) a.elements[i][j] = log(0);
		}
	}
	
	return a;
}

matrix get_emissmat(dictionary* dic, matrix* x, int labels[MAX_N_WORD]) {
	matrix b;
	int T, state_index, n_state;
	
	T = x->rows + 2;
	state_index = 1;
	n_state = labels[MAX_N_WORD - 1] + 2;
	b = new_matrix(n_state, T);
	
	for(int w = 0; w < dic->size; w++) {
		for(int p = 0; p < dic->words[w].n_phone; p++) {
			for(int s = 0; s < hmm[dic->words[w].hmm_indices[p]].n_state; s++) {
				for(int t = 1; t < T - 1; t++) {
					b.elements[state_index][t] = emission(dic->words[w].hmm_indices[p], s, x, t - 1);
				}
				
				state_index++;
			}
		}
	}
	
	return b;
}

void recognize(char* indir, char* outfile) {
	FILE* f;
	if((f = fopen(outfile, "w")) != NULL) {
		fprintf(f, "#!MLF!#\n");
		
		dictionary dic = get_dictionary("dictionary.txt");
		unigram ug = get_unigram(&dic, "unigram.txt");
		bigram bg = get_bigram(&dic, "bigram.txt");
		
		int* labels = get_labels(&dic);
		
		matrix a = get_transmat(&dic, &ug, &bg, labels);
		
		test_inputs(indir, &dic, labels, &a, f);

		free_matrix(&a);
		free(labels);
		fclose(f);
	} else {
		printf("error: cannot open %s\n", outfile);
	}
}

dictionary get_dictionary(char* filename) {
	dictionary dic;
	FILE* f;
	char wordbuf[MAX_L_WORD];
	char phobuf[16];
	
	dic.size = 0;
	dic.n_word = 0;
	if((f = fopen(filename, "r")) != NULL) {
		while(fscanf(f, "%s\t%[0-9a-zA-Z ]", wordbuf, phobuf) > 0) {
			word w;
			char *tok;
			
			w.n_phone = 0;
			tok = strtok(phobuf, " ");
			
			strcpy(w.str, wordbuf);
			
			int idx = word_index(&dic, w.str);
			if(idx != -1) {
				w.index = idx;
			} else {
				w.index = dic.size;
				dic.n_word++;
			}
			
			while(tok != NULL) {
				strcpy(w.phones[w.n_phone], tok);
				for(int h = 0; h < MAX_N_HMM; h++) {
					if(!strcmp(w.phones[w.n_phone], hmm[h].name))
						w.hmm_indices[w.n_phone] = h;
				}
				w.n_phone++;
				tok = strtok(NULL, " ");
			}
			dic.words[dic.size++] = w;
		}
		
		fclose(f);
	} else {
		printf("error: cannot open %s\n", filename);
	}

	return dic;
}

int word_index(dictionary* dic, char* w) {
	for(int i = 0; i < dic->size; i++) {
		if(!strcmp(dic->words[i].str, w))
			return dic->words[i].index;
	}
	
	return -1;
}

void init_unigram(unigram* ug) {
	ug->size = 0;
	for(int i = 0; i < MAX_N_WORD; i++) {
		ug->prob[i] = 0;
	}
}

void init_bigram(bigram* bg) {
	bg->size = 0;
	for(int i = 0; i < MAX_N_WORD; i++) {
		for(int j = 0; j < MAX_N_WORD; j++) {
			bg->prob[i][j] = 0;
		}
	}
}

unigram get_unigram(dictionary* dic, char* filename) {
	unigram ug;
	FILE* f;
	char wordbuf[MAX_L_WORD];
	double prob;
	
	init_unigram(&ug);
	
	if((f = fopen(filename, "r")) != NULL) {
		while(fscanf(f, "%s\t%lf", wordbuf, &prob) > 0) {
			ug.prob[word_index(dic, wordbuf)] = prob;
			ug.size++;
		}
		
		fclose(f);
	} else {
		printf("error: cannot open %s\n", filename);
	}
	return ug;
}

bigram get_bigram(dictionary* dic, char* filename) {
	bigram bg;
	FILE* f;
	char wordbuf1[MAX_L_WORD];
	char wordbuf2[MAX_L_WORD];
	double prob;
	
	init_bigram(&bg);
	
	if((f = fopen(filename, "r")) != NULL) {
		while(fscanf(f, "%s\t%s\t%lf", wordbuf1, wordbuf2, &prob) > 0) {
			
			bg.prob[word_index(dic, wordbuf1)][word_index(dic, wordbuf2)] = prob;
			bg.size++;
		}
		
		fclose(f);
	} else {
		printf("error: cannot open %s\n", filename);
	}
	return bg;
}

void test_inputs(char* dirname, dictionary* dic, int labels[MAX_N_WORD], matrix* a, FILE* f) {
	DIR* d;
	struct dirent *dir;
	char path[L_BUFFER];
	
	memset(path, '\0', 1);
	d = opendir(dirname);
	if(d) {
		while((dir = readdir(d)) != NULL) {
			if(dir->d_name[0] != '.') {
				strcat(path, dirname);

				if(dirname[strlen(dirname) - 1] != '\\')
					strcat(path, "\\");

				strcat(path, dir->d_name);

				if (dir->d_type == DT_DIR){
					test_inputs(path, dic, labels, a, f);
				} else {
					if(f != NULL) {
						
						printf("%s\n", path);
						matrix x = read_input(path);
						
						// get <results>
						matrix b = get_emissmat(dic, &x, labels);
						int* q = viterbi(a, &b);
												
						remove_ext(&path);
						strcat(path, ".rec");
						fprintf(f, "\"%s\"\n", path);
						
						// fprintf <results>
						print_results(f, dic, q, b.cols - 1, labels);
						
						free_matrix(&x);
						free_matrix(&b);
						free(q);
						
						fprintf(f, ".\n");
					} else {
						printf("error: null file descriptor");
					}
				}
				
				memset(path, '\0', 1);
			}
		}
		
		closedir(d);
	}
}

matrix new_matrix(int m, int n) {
	matrix mat;
	mat.rows = m;
	mat.cols = n;
	mat.elements = (double**)malloc(mat.rows * sizeof(double*));
	
	for(int i = 0; i < mat.rows; i++){
		mat.elements[i] = (double*)malloc(mat.cols * sizeof(double));
	}
	
	for(int i = 0; i < mat.rows; i++)
		for(int j = 0; j < mat.cols; j++)
			mat.elements[i][j] = 0;
	
	return mat; 
}

matrix read_input(char* filename) {
	FILE* f;
	double** mat;
	matrix ret;
	
	if((f = fopen(filename, "r")) != NULL) {

		fscanf(f, "%d %d\n", &ret.rows, &ret.cols);
		
		ret = new_matrix(ret.rows, ret.cols);

		for(int r = 0; r < ret.rows; r++) {
			for(int c = 0; c < ret.cols; c++) {
				fscanf(f, "%lf", &ret.elements[r][c]);
			}
		}
		
		fclose(f);
	} else {
		printf("error: cannot open %s\n", filename);
	}
	return ret;
}

void remove_ext(char (*str)[]) {
	int i, len;
	
	len = strlen(*str);
	for(i = len - 1; i >= 0; i--) {
		if((*str)[i] == '.') {
			(*str)[i] = '\0';
			break;
		}
	}
}

void print_matrix(matrix* mat) {
	for(int i = 0; i < mat->rows; i++) {
		for(int j = 0; j < mat->cols; j++) {
			printf("%lf", mat->elements[i][j]);
			if(j == mat->cols - 1) {
				printf("\n");
			} else {
				printf(", ");
			}
		}
	}
}

void free_matrix(matrix* mat) {
	for(int i = 0; i < mat->rows; i++) {
		free(mat->elements[i]);
	}
	free(mat->elements);
	mat->elements = NULL;
}
