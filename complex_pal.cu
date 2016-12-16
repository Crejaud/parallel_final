#include <cuda.h>
#include <cuda_runtime_api.h>
#include <string>
#include <vector>

// substrings
__global__ void get_all_substrings_cuda(string word) {
  unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;

  // check for out of bounds
  if (i >= word.size() || j > word.size() - i) {
    return;
  }

  // got substring
  string substring = word.substr(i, j);
}


// palindromes
__global__ void find_palindromes_of_anagrams_cuda(string double_letters, string single, int num_perms) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  string perm = substring;

  if (i >= num_perms) {
    return;
  }

  for (int b = double_letters.size(), div = num_perms, b > 0; b--) {
    div/=b;
    int index = (i/div)%b;
    // perm is a permutation!
    // now mirror over!

    perm.erase(index,1);
  }
}

void kernel_substring_wrapper(string word) {
  dim3 threads_per_block(64, 64);
  dim3 blocks_per_dimension(16, 16);
  get_all_substrings_cuda<<<blocks_per_dimension, threads_per_block>>>(string word);
}

void kernel_palindrome_wrapper(vector<string> substrings) {
  for (int sub_index = 0; i < substrings.size(); i++) {
    string substring = substrings[i];
    int frequencies[26] = {0};
    int i;
    int size_of_substring = substring.size();
    string single = "";
    string double_letters = "";
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
        if (single.empty()) {
          single = i+'A';
        } else {
          // there are no palindromes of any anagram of this substring
          return;
        }
      }
    }

    int num_perms = 1;
    int size_of_double_letters = double_letters.size();
    // get the number of permuations
    for (i=1; i<=size_of_double_letters; num_perms*=i++);

    dim3 threads_per_block(64, 64);
    dim3 blocks_per_dimension(16, 16);
    find_palindromes_of_anagrams_cuda<<<blocks_per_dimension, threads_per_block>>>(double_letters, single, num_perms);
  }
}
