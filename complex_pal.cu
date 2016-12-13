#include <iostream>
#include <vector>
#include <string>
#include <ctime>
#include <thread>
#include <mutex>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "complex_pal.h"
using namespace std;

static const char capital_letters[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";

int main() {
  vector<string> palindromes_seq;
  vector<string> substrings_seq;

  vector<string> palindromes_par;
  vector<string> substrings_par;

  mutex mtx;

  clock_t start, end;
  double sub_duration_seq, pal_duration_seq, sub_duration_par, pal_duration_par, pal_duration_cuda;
  int word_length;
  string word = "";
  int i,j;
  cout << "Please enter the integer size of your word: ";
  cin >> word_length;

  int rand_num;

  if (word_length <= 0) {
    cout << "Invalid size. Defaulting to size 1024." << endl;
    word_length = 1024;
  }

  if (word_length % 2 == 1) {
    rand_num = rand() % (sizeof(capital_letters) - 1);
    word += capital_letters[rand_num];
  }

  for (i = 0; i < word_length/2; i++) {
    rand_num = rand() % (sizeof(capital_letters) - 1);
    word += capital_letters[rand_num];
    word += capital_letters[rand_num];
  }

  cout << "[Sequential] Computing substrings." << endl;

  // start clock for getting substrings.
  start = clock();

  get_all_substrings_seq(word, substrings_seq);

  end = clock();

  sub_duration_seq = (end - start)/(double) CLOCKS_PER_SEC;

  cout << "[Sequential] Found " << substrings_seq.size() << " substrings." << endl;

  cout << "[Parallel] Computing substrings." << endl;

  start = clock();

  get_all_substrings_par(word, substrings_par);

  end = clock();

  sub_duration_par = (end - start)/(double) CLOCKS_PER_SEC;

  cout << "[Parallel] Found " << substrings_par.size() << " substrings." << endl;

  cout << "[Sequential] Computing palindrome existance for each substring." << endl;

  start = clock();

  // sequential
  for (auto& substring: substrings_seq) {
    //cout << "Finding palindromes in anagrams of " << substring << endl;
    find_palindromes_of_anagrams(substring, palindromes_seq, 0);
  }

  end = clock();

  pal_duration_seq = (end - start)/(double) CLOCKS_PER_SEC;

  cout << "----------------------------------------------------------" << endl;
  cout << "----------------------------------------------------------" << endl;
  cout << "[Sequential] Found " << palindromes_seq.size() << " palindromes." << endl;
  cout << "[Sequential] Finding substrings: " << sub_duration_seq << " clock cycles" << endl;
  cout << "[Sequential] Finding existance of palindromes of anagrams of substrings took: " << pal_duration_seq << " clock cycles" << endl;
  cout << "[Sequential] Total duration: " << sub_duration_seq + pal_duration_seq << " clock cycles" << endl;
  cout << "----------------------------------------------------------" << endl;
  cout << "----------------------------------------------------------" << endl;

  cout << "[Parallel] Computing palindrome existance for each substring." << endl;

  start = clock();

  // parallel
  int num_threads;
  cout << "How many threads would you like to run? [1-8] ";
  cin >> num_threads;

  if (num_threads > substrings_par.size())
    num_threads = substrings_par.size();

  thread *myThreads = new thread[num_threads];

  // span it
  int span = substrings_par.size()/num_threads;
  int first, last;

  for (i = 0; i < num_threads; i++) {
    first = i*span;
    last = (i+1)*span;

    if (i == num_threads - 1)
      last = substrings_par.size();

    //cout << first << ", " << last << ", " << span << endl;

    myThreads[i] = thread(find_palindromes_of_anagrams_par, substrings_par, palindromes_par, first, last);
  }

  for (j = 0; j < num_threads; j++) {
    myThreads[j].join();
  }

  //cout << "Freeing myThreads" << endl;
  delete [] myThreads;

  end = clock();

  pal_duration_par = (end - start)/(double) CLOCKS_PER_SEC;

  cout << "----------------------------------------------------------" << endl;
  cout << "----------------------------------------------------------" << endl;
  cout << "[Parallel] Found " << palindromes_par.size() << " palindromes." << endl;
  cout << "[Parallel] Finding substrings: " << sub_duration_par << " clock cycles" << endl;
  cout << "[Parallel] Finding existance of palindromes of anagrams of substrings took: " << pal_duration_par << " clock cycles" << endl;
  cout << "[Parallel] Total duration: " << sub_duration_par + pal_duration_par << " clock cycles" << endl;
  cout << "----------------------------------------------------------" << endl;
  cout << "----------------------------------------------------------" << endl;

  char** substrings_cuda;
  char** palindromes_cuda;
  int num_subs = substrings_seq.size();
  int num_pals = palindromes_seq.size();

  vector<char*> temp_substrings_cuda(substrings_seq.size());
  for(unsigned i = 0; i < substrings_seq.size(); ++i)
  {
      temp_substrings_cuda[i] = substrings_seq[i].data();
  } //you can use transform instead of this loop
  char** temp_temp_substrings_cuda = temp_substrings_cuda.data();

  // copy into cuda substrings
  cudaMallocManaged(&substrings_cuda, palindromes_seq.size() * sizeof(char *));
  for (unsigned i = 0; i < substrings_seq.size(); i++) {
    cudaMallocManaged(&(substrings_cuda[i]), temp_temp_substrings_cuda[i] * sizeof(char));
  }
  // initialize cuda substrings
  for (unsigned i = 0; i < substrings_seq.size(); i++) {
    strcpy(substrings_cuda[i], temp_temp_substrings_cuda[i]);
  }

  cudaEvent_t	start_gpu, stop_gpu;
	cudaEventCreate(&start_gpu);
	cudaEventCreate(&stop_gpu);
	cudaEventRecord(start_gpu, 0);

  kernel_palindrome_wrapper(substrings_cuda, substrings_seq.size());

  cudaEventRecord(stop_gpu, 0);
	cudaEventSynchronize(stop_gpu);

	double gpu_elapsed_time;
	cudaEventElapsedTime(&gpu_elapsed_time, start_gpu, stop_gpu);
	printf( "Time for GPU To Compute Product: %.5f s\n", gpu_elapsed_time/1000.0f );
	cudaEventDestroy(start_gpu);
	cudaEventDestroy(stop_gpu);

  return 0;
}

void get_all_substrings_seq(string word, vector<string> substrings) {
  int i, j;
  for (i = 0; i < word.size(); i++) {
    for (j = 1; j <= word.size() - i; j++) {
      substrings.push_back(word.substr(i, j));
    }
  }
}

void get_all_substrings_par(string word, vector<string> substrings) {
  int i, j;
  int num_threads;
  cout << "How many threads would you like to run? ";
  cin >> num_threads;

  if (num_threads > word.size())
    num_threads = word.size();

  thread *myThreads = new thread[num_threads];

  // span it
  int span = word.size()/num_threads;
  int first, last;

  for (i = 0; i < num_threads; i++) {
    first = i*span;
    last = (i+1)*span;

    if (i == num_threads - 1)
      last = word.size();

    //cout << first << ", " << last << ", " << span << endl;
    myThreads[i] = thread(get_all_substrings_par_thread, word, substrings, first, last);
  }

  //cout << "Waiting on threads." << endl;
  for (j = 0; j < num_threads; j++) {
    //cout << "Waiting for thread " << j << endl;
    myThreads[j].join();
  }
}

void get_all_substrings_par_thread(string word, vector<string> substrings, int start, int end) {
  int j;
  for (; start < end; start++) {
    for (j = 1; j <= word.size() - start; j++) {
      mtx.lock();
      substrings.push_back(word.substr(start, j));
      mtx.unlock();
    }
  }
}

void find_palindromes_of_anagrams(string substring, vector<string> palindromes, int flag) {
  // initialize character array to 26 (letters in the alphabet)
  int frequencies[26] = {0};
  int i;
  string single = "";
  string double_letters = "";
  // gets frequencies of each character in the substring.
  for (i = 0; i < substring.size(); i++) {
    int ascii_val = substring[i] - 'A';
    frequencies[ascii_val]++;
    if (frequencies[ascii_val] > 1 && frequencies[ascii_val] % 2 == 0) {
      double_letters += substring[i];
    }
  }
  // can't make a palindrome.
  if (double_letters.empty()) {
    return;
  }

  for (i = 0; i < 26; i++) {
    if (frequencies[i] >= 1 && frequencies[i] % 2 == 1) {
      if (single.empty()) {
        single = i+'A';
      } else {
        // there are no palindromes of any anagram of this substring
        return;
      }
    }
  }

  // now we must find all palindromes of all anagrams of this substring!
  recursive_palindrome_anagram_finder("", single, double_letters, palindromes, flag);
}

void find_palindromes_of_anagrams_par(vector<string> substrings, vector<string> palindromes, int start, int end) {
  for (; start < end; start++) {
    //cout << "Finding palindromes in anagrams of " << substrings_par[start] << endl;
    find_palindromes_of_anagrams(substrings[start], palindromes, 1);
  }
}

void recursive_palindrome_anagram_finder(string pre, string single, string double_letters, vector<string> palindromes, int flag) {
  int i;

  if (double_letters.empty()) {
    int pre_length_before_single = pre.size();
    pre += single;
    // now we mirror it over
    for (i = pre_length_before_single - 1; i >= 0; i--) {
      pre += pre[i];
    }
    // now add pre to list of palindromes_seq because this is a palindrome
    if (flag == 0)
      palindromes.push_back(pre);
    else {
      mtx.lock();
      palindromes.push_back(pre);
      mtx.unlock();
    }
    return;
  }

  for (i = 0; i < double_letters.size(); i++) {
    recursive_palindrome_anagram_finder(pre + double_letters[i],
      single,
      double_letters.substr(0, i) + double_letters.substr(i+1),
      palindromes
      flag);
  }
}

// substrings
__global__ void get_all_substrings_cuda(char* word, int length) {
  unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;

  // check for out of bounds
  if (i >= length || j > length - i) {
    return;
  }

  // got substring
}


// palindromes
__global__ void find_palindromes_of_anagrams_cuda(char** double_letters, int double_letters_size, char single, int num_perms) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  char** perm = malloc(strlen(double_letters) + 1);
  strcpy(perm, double_letters);

  if (i >= num_perms) {
    return;
  }

  for (int b = double_letters_size, div = num_perms; b > 0; b--) {
    div/=b;
    int index = (i/div)%b;
    // perm is a permutation!
    // now mirror over!

    memmove(&perm[index], &perm[index+1], strlen(perm) - index);
  }
}

void kernel_substring_wrapper(char* word, int length) {
  dim3 threads_per_block(8, 8);
  dim3 blocks_per_dimension(length/8, length/8);
  get_all_substrings_cuda<<<blocks_per_dimension, threads_per_block>>>(word);
}

void kernel_palindrome_wrapper(char** substrings, int num_substrings) {
  for (int sub_index = 0; sub_index < num_substrings; sub_index++) {
    char** substring = substrings[sub_index];
    int frequencies[26] = {0};
    int i;
    int size_of_substring = sizeof(substring)/sizeof(char);
    char single = '\0';
    char** double_letters = "";
    // gets frequencies of each character in the substring.
    for (i = 0; i < size_of_substring; i++) {
      int ascii_val = substring[i] - 'A';
      frequencies[ascii_val]++;
      if (frequencies[ascii_val] > 1 && frequencies[ascii_val] % 2 == 0) {
        double_letters += substring[i];
      }
    }
    // can't make a palindrome.
    if (double_letters.empty()) {
      return;
    }

    for (i = 0; i < 26; i++) {
      if (frequencies[i] >= 1 && frequencies[i] % 2 == 1) {
        if (single == '\0') {
          single = i+'A';
        } else {
          // there are no palindromes of any anagram of this substring
          return;
        }
      }
    }

    int num_perms = 1;
    int size_of_double_letters = sizeof(double_letters)/sizeof(char);
    // get the number of permuations
    for (i=1; i<=size_of_double_letters; num_perms*=i++);

    dim3 threads_per_block(16);
    dim3 blocks_per_dimension(num_perms/16);
    find_palindromes_of_anagrams_cuda<<<blocks_per_dimension, threads_per_block>>>(double_letters, size_of_double_letters, single, num_perms);
  }
}
