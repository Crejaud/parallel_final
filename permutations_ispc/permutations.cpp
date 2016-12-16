#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <immintrin.h>
#include "permutations_ispc.h"

using namespace ispc;

//initialize functions & globals
int* setupWord(int);

int main()
{
	clock_t begin, end;
	double timeSpent;
  int word_length = 11;
  int* word = setupWord(word_length);

  uint64_t num_perm = 1;
  for (int k = 1;k <= word_length; num_perm *= k++);
  // get the number of permutations
  printf("word: ");
  for (int i = 0; i < word_length; i++) {
    printf("%d", word[i]);
  }
  printf("\n");
  printf("length: %d\n", word_length);
  printf("number of permutations: %d\n", num_perm);
	// CALLING PERM IN ISPC
	int num_cores;
	int num_threads;
	// NO TASKS: 1-8 threads
	for (num_threads = 1; num_threads <= 8; num_threads++)
	{
		begin = clock();
		perm_ispc(word, num_perm, num_threads);
		end = clock();
		timeSpent = (double)(end - begin) / CLOCKS_PER_SEC;
		printf("[ISPC] 1 core | %d thread(s) | %f cycles\n", num_threads, timeSpent);
	}

	// TASKS: 2-4 cores & 1-8 threads
	for (num_cores = 2; num_cores <= 4; num_cores++)
	{
		for (num_threads = 1; num_threads <= 8; num_threads++)
		{
			begin = clock();
			perm_ispc_tasks(word, num_perm, num_threads, num_cores);
			end = clock();
			timeSpent = (double)(end - begin) / CLOCKS_PER_SEC;
			printf("[ISPC] %d cores | %d thread(s) | %f cycles\n",
				num_cores,
				num_threads,
				timeSpent);
		}
	}
  free(word);

	return 0;
}

// create word of that length
int* setupWord(int word_length) {

  int* word;
  int rand_num;

  word = (int*) malloc(word_length * sizeof(int));
  int index = 0;
  if (word_length % 2 == 1) {
    rand_num = rand() % (10);
    word[index++] = rand_num;
  }

  for (int i = 0; i < word_length/2; i++) {
    rand_num = rand() % (10);
    word[index++] = rand_num;
    word[index++] = rand_num;
  }

  return word;
}
