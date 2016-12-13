#include <iostream>
#include <vector>
#include <string>
#include <ctime>
#include <thread>
#include <mutex>
using namespace std;

void find_permutations_rec(string);
void find_permutations_rec_helper(string, string);
void find_permutations_par(string, int);
void find_permutations(string, long long, long long, long long);

string setupWord(int);

static const char capital_letters[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";

int main() {
  clock_t start, end;
  double duration;
  int word_length, num_threads;
  string word = "";
  cout << "Please enter the integer size of your word: ";
  cin >> word_length;

  // create word that is nice and consistent for this problem
  word = setupWord(word_length);

  start = clock();
  // do sequential recursive
  find_permutations_rec(word);
  end = clock();

  duration = (end - start)/(double) CLOCKS_PER_SEC;

  cout << "[Sequential - Recursive] Permutations: " << duration << " clock cycles" << endl;

  start = clock();
  long long perm=1, digits=word.size();
  for (int k=1;k<=digits;perm*=k++);
  cout << digits << " " << perm << endl;
  cout << word << endl;
  find_permutations(word, 0, perm, perm);

  end = clock();

  duration = (end - start)/(double) CLOCKS_PER_SEC;

  cout << "[Sequential - Iterative] Permutations: " << duration << " clock cycles" << endl;







  cout << "How many threads would you like to run? [1-8] ";
  cin >> num_threads;

  start = clock();
  // do parallel iterative
  find_permutations_par(word, num_threads);

  end = clock();

  duration = (end - start)/(double) CLOCKS_PER_SEC;

  cout << "[Parallel - Iterative] Permutations: " << duration << " clock cycles" << endl;

  return 0;
}

void find_permutations_par(string word, int num_threads) {

  long long perm = 1, digits = word.size();
  for (int k = 1;k <= digits; perm *= k++);

  if (num_threads > word.size())
    num_threads = word.size();

  thread *myThreads = new thread[num_threads];

  // span it
  long long span = perm/num_threads;
  long long first, last;

  first = (num_threads - 1) * span;
  last = perm;

  find_permutations(word, first, last, perm);

  for (int i = 0; i < num_threads - 1; i++) {
    first = i*span;
    last = (i+1)*span;

    if (i == num_threads - 1)
      last = perm;

    //cout << first << ", " << last << ", " << span << endl;

    myThreads[i] = thread(find_permutations, word, first, last, perm);
  }

  for (int j = 0; j < num_threads - 1; j++) {
    myThreads[j].join();
  }

  //cout << "Freeing myThreads" << endl;
  delete [] myThreads;
}

void find_permutations(string word, long long first, long long last, long long num_perm) {
 for (;first<last;first++) {
    string temp = word;

    long long div = num_perm;
    string perm = "";
    for (int digit = word.size(); digit > 0; digit--)
    {
      // compute the number of repetitions for one character in the actual column
      div /= digit;
      //compute the index of the character in the string
      long long t = first / div;
      int index = t % digit;
      perm += temp[index];
      //remove the used character
      temp.erase(index,1) ;
    }
  }
}

// create word of that length
string setupWord(int word_length) {
  string word;
  int rand_num;

  if (word_length <= 0) {
    cout << "Invalid size. Defaulting to size 1024." << endl;
    word_length = 1024;
  }

  if (word_length % 2 == 1) {
    rand_num = rand() % (sizeof(capital_letters) - 1);
    word += capital_letters[rand_num];
  }

  for (int i = 0; i < word_length/2; i++) {
    rand_num = rand() % (sizeof(capital_letters) - 1);
    word += capital_letters[rand_num];
    word += capital_letters[rand_num];
  }

  return word;
}


void find_permutations_rec(string word) {
  find_permutations_rec_helper("", word);
}

void find_permutations_rec_helper(string pre, string post) {
  if (post.empty()) {
    // found palindrome since pre is a palindrome
    return;
  }
  for (int i = 0; i < post.size(); i++) {
    find_permutations_rec_helper(pre + post[i],
      post.substr(0, i) + post.substr(i+1));
  }
}
