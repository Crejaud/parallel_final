#include <iostream>
#include <vector>
#include <string>
#include <ctime>
#include <thread>
#include "complex_pal.h"
using namespace std;

static const char capital_letters[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";

vector<string> palindromes_seq;
vector<string> substrings_seq;

vector<string> palindromes_par;
vector<string> substrings_par;

int main() {
  clock_t start, end;
  double sub_duration_seq, pal_duration_seq, sub_duration_par, pal_duration_par;
  int word_length;
  string word = "";
  int i;
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

  get_all_substrings_seq(word);

  end = clock();

  sub_duration_seq = (end - start)/(double) CLOCKS_PER_SEC;

  cout << "[Sequential] Found " << substrings_seq.size() << " substrings." << endl;

  cout << "[Parallel] Computing substrings." << endl;

  start = clock();

  get_all_substrings_par(word);

  end = clock();

  sub_duration_par = (end - start)/(double) CLOCKS_PER_SEC;

  cout << "[Parallel] Found " << substrings_par.size() << " substrings." << endl;

  cout << "[Sequential] Computing palindrome existance for each substring." << endl;

  start = clock();

  // sequential
  for (auto& substring: substrings_seq) {
    cout << "Finding palindromes in anagrams of " << substring << endl;
    find_palindromes_of_anagrams(substring, 0);
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
  i = 0;
  int num_threads;
  cout << "How many threads would you like to run? [1-8] ";
  cin >> num_threads;

  thread *myThreads = new thread[num_threads];
  for (auto& substring: substrings_par) {
    if (i >= num_threads) {
      for (i = 0; i < num_threads; i++) {
        myThreads[i].join();
      }
      i = 0;
    }
    cout << "Finding palindromes in anagrams of " << substring << endl;
    myThreads[i] = thread(find_palindromes_of_anagrams, substring, 1);
    i++;
  }

  for (i = 0; i < num_threads; i++) {
    myThreads[i].join();
  }

  delete [] myThreads;

  end = clock();

  pal_duration_par = (end - start)/(double) CLOCKS_PER_SEC;

  cout << "----------------------------------------------------------" << endl;
  cout << "----------------------------------------------------------" << endl;
  cout << "[Parallel] Found " << palindromes_par.size() << " palindromes." << endl;
  cout << "[Parallel] Finding substrings: " << sub_duration_par << "clock cycles" << endl;
  cout << "[Parallel] Finding existance of palindromes of anagrams of substrings took: " << pal_duration_par << " clock cycles" << endl;
  cout << "[Parallel] Total duration: " << sub_duration_par + pal_duration_par << " clock cycles" << endl;
  cout << "----------------------------------------------------------" << endl;
  cout << "----------------------------------------------------------" << endl;

  cout << "Here are the palindromes that I found:" << endl;

  return 0;
}

void get_all_substrings_seq(string word) {
  int i, j;
  for (i = 0; i < word.size(); i++) {
    for (j = 1; j <= word.size() - i; j++) {
      substrings_seq.push_back(word.substr(i, i+j));
    }
  }
}

void get_all_substrings_par(string word) {
  int i;
  thread *myThreads = new thread[word.size()];
  for (i = 0; i < word.size(); i++) {
    // create thread
    myThreads[i] = thread(get_all_substrings_par_thread, word, i);
  }

  for (i = 0; i < word.size(); i++) {
    myThreads[i].join();
  }

  delete [] myThreads;
}

void get_all_substrings_par_thread(string word, int i) {
  int j;
  for (j = 1; j <= word.size() - i; j++) {
    substrings_par.push_back(word.substr(i, i+j));
  }
}

void find_palindromes_of_anagrams(string substring, int flag) {
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
  recursive_palindrome_anagram_finder("", single, double_letters, flag);
}

void recursive_palindrome_anagram_finder(string pre, string single, string double_letters, int flag) {
  int i;

  if (double_letters.empty()) {
    int pre_length_before_single = pre.size();
    pre += single;
    // now we mirror it over
    for (i = pre_length_before_single - 1; i >= 0; i++) {
      pre += pre[i];
    }
    // now add pre to list of palindromes_seq because this is a palindrome
    if (flag == 0)
      palindromes_seq.push_back(pre);
    else
      palindromes_par.push_back(pre);
    return;
  }

  for (i = 0; i < double_letters.size(); i++) {
    recursive_palindrome_anagram_finder(pre + double_letters[i],
      single,
      double_letters.substr(0, i) + double_letters.substr(i+1, double_letters.size()),
      flag);
  }

}
