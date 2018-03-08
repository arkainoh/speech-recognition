#include "hmm_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void test_emissmat() {
	dictionary dic = get_dictionary("dictionary.txt");
	unigram ug = get_unigram(&dic, "unigram.txt");
	bigram bg = get_bigram(&dic, "bigram.txt");
	matrix x = read_input("tst\\f\\ak\\1237743.txt");
	int* labels = get_labels(&dic);
	matrix a = get_transmat(&dic, &ug, &bg, labels);
	matrix b = get_emissmat(&dic, &x, labels);
	free_matrix(&a);
	free_matrix(&b);
	free_matrix(&x);
	free(labels);
}

void test_unigram(dictionary* dic) {
	unigram ug = get_unigram(dic, "unigram.txt");
	for(int i = 0; i < dic->n_word; i++) {
		printf("%s\t%lf\n", dic->words[i].str, ug.prob[i]);
	}
	printf("size: %d\n", ug.size);
}

void test_bigram(dictionary* dic) {
	bigram bg = get_bigram(dic, "bigram.txt");
	for(int i = 0; i < dic->n_word; i++) {
		for(int j = 0; j < dic->n_word; j++) {
			printf("%s\t%s\t%lf\n", dic->words[i].str, dic->words[j].str, bg.prob[i][j]);
		}
	}
	printf("size: %d\n", bg.size);
}

void test_dic() {
	dictionary dic = get_dictionary("dictionary.txt");
	for(int i = 0; i < dic.size; i++) {
		printf("%d: %s\t", dic.words[i].index, dic.words[i].str);
		for(int j = 0; j < dic.words[i].n_phone; j++) {
			printf("%s(%d) ", dic.words[i].phones[j], dic.words[i].hmm_indices[j]);
		}
		printf("\n");
	}
	printf("size: %d\n", dic.size);
	printf("n_word: %d\n", dic.n_word);
}

void test_inf() {
	double inf = log(0);
	printf("%lf < 0 = %d\n", inf, inf < 0);
	printf("%lf > 0 = %d\n", inf, inf > 0);
	printf("%lf == 0 = %d\n", inf, inf == 0);
	printf("%lf == %lf = %d\n", inf, inf, inf == inf);
	printf("%lf == log(0) = %d\n", inf, inf - 52 == log(0));
	printf("%lf + 32 = %lf\n", inf, inf + 32);
	printf("%lf - 32 = %lf\n", inf, inf - 32);
	printf("%lf + %lf = %lf\n", inf, inf, inf + inf);
	printf("%lf - %lf = %lf\n", inf, inf, inf - inf);
	printf("%lf * %lf = %lf\n", inf, inf, inf * inf);
	printf("%lf / %lf = %lf\n", inf, inf, inf / inf);
}

void test_a() {
	dictionary dic = get_dictionary("dictionary.txt");
	unigram ug = get_unigram(&dic, "unigram.txt");
	bigram bg = get_bigram(&dic, "bigram.txt");
	int* labels = get_labels(&dic);
	matrix a = get_transmat(&dic, &ug, &bg, labels);
	print_matrix(&a);
	free_matrix(&a);
	free(labels);
}

void test_b() {
	matrix x = read_input("tst\\f\\ak\\1237743.txt");
	for(int i = 0; i < x.rows; i++) {
		printf("bs(x) = %.18lf\n", emission(19, 2, &x, i));
	}
	free_matrix(&x);
}

void test_viterbi() {
	matrix a = new_matrix(4, 4);
	matrix b = new_matrix(4, 6);
	
	a.elements[0][1] = 0.6;
	a.elements[0][2] = 0.4;
	a.elements[1][1] = 0.7;
	a.elements[1][2] = 0.3;
	a.elements[2][1] = 0.4;
	a.elements[2][2] = 0.6;
	
	b.elements[1][1] = 0.1;
	b.elements[1][2] = 0.1;
	b.elements[1][3] = 0.5;
	b.elements[1][4] = 0.4;
	b.elements[2][1] = 0.6;
	b.elements[2][2] = 0.6;
	b.elements[2][3] = 0.1;
	b.elements[2][4] = 0.3;
	
	for(int i = 0; i < 4; i++) {
		for(int j = 0; j < 4; j++) {
			a.elements[i][j] = log(a.elements[i][j]);
			if(b.elements[i][j] != 0)
				b.elements[i][j] = log(b.elements[i][j]);
		}
	}
	
	int* q = viterbi(&a, &b); // answer: 2 2 1 1
	
	print_matrix(&a);
	printf("\n");
	print_matrix(&b);
	printf("\n");
	
	for(int t = 0; t < b.cols; t++) {
		printf("%d ", q[t]);
	}
	printf("\n");
	
	free_matrix(&a);
	free_matrix(&b);
	free(q);
}

void test_recognize(char* path) {
	dictionary dic = get_dictionary("dictionary.txt");
	unigram ug = get_unigram(&dic, "unigram.txt");
	bigram bg = get_bigram(&dic, "bigram.txt");
	matrix x = read_input(path);
	
	int* labels = get_labels(&dic);
	
	matrix a = get_transmat(&dic, &ug, &bg, labels);
	matrix b = get_emissmat(&dic, &x, labels);
	
	int* q = viterbi(&a, &b);
	printf("test: %s\n\n", path);
	printf("<labels>\n");
	printf("%s: %d ~ %d\n", dic.words[0].str, 1, labels[0]);
	
	for(int w = 1; w < dic.size; w++) {
		printf("%s: %d ~ %d\n", dic.words[w].str, labels[w - 1] + 1, labels[w]);
	}
	printf("\n");
	
	printf("<state sequence>\n");
	for(int t = 0; t < b.cols - 1; t++) {
		printf("%d ", q[t]);
	}
	printf("\n");
	
	print_results(NULL, &dic, q, b.cols - 1, labels);
	
	free_matrix(&a);
	free_matrix(&b);
	free_matrix(&x);
	free(labels);
	free(q);
}

int main() {

	// dictionary dic = get_dictionary("dictionary.txt");
	// unigram ug = get_unigram(&dic, "unigram.txt");
	// bigram bg = get_bigram(&dic, "bigram.txt");
	// test_viterbi();
	// test_recognize();
	// test_a();
	// test_b();
	// test_dic();
	// get_transmat(&dic, &ug, &bg);
	// test_recognize("tst\\f\\ak\\1237743.txt");
		
	recognize("tst", "recognized.txt");
	
	return 0;
}