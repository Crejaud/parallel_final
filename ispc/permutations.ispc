const uniform int word_length = 10;

static void find_permutation(uniform int word[], uniform unsigned int64 num_perm, unsigned int64 i)
{
  // copy word onto temp
  int temp[word_length];
  int temp_length = word_length;
  for (int k = 0; k < word_length; k++) {
    temp[k] = word[k];
  }

  unsigned int64 div = num_perm;
  int perm[word_length];
  int perm_index = 0;
  for (unsigned int64 digit = word_length; digit > 0; digit--)
  {
    div /= digit;
    unsigned int64 t = i / div;
    int index = t % digit;
    perm[perm_index] = temp[index];
    perm_index++;

    // shift digits over
    for (int j = index; j < temp_length; j++) {
      temp[j] = temp[j+1];
    }
    //memmove(temp+index, temp+index+1, temp_length - index);
    temp_length--;
  }
}

static void perm_no_task(uniform int word[], uniform unsigned int64 num_perm)
{
	foreach (i = 0 ... num_perm)
	{
		find_permutation(word, num_perm, i);
	}
}

static void perm_t_no_task(uniform int word[], uniform unsigned int64 num_perm , uniform int num_threads)
{
	foreach (i = 0 ... num_perm)
  {
    unsigned int64 first_t, last_t, span_t;
    span_t = num_perm / num_threads;
    first_t = i * span_t;
    last_t = min((i+1) * span_t, num_perm);

		unsigned int64 i_t;
		for (i_t = first_t; i_t < last_t; i_t++)
		{
      find_permutation(word, num_perm, i_t);
		}
	}
}

static task void perm_task(uniform int word[], uniform unsigned int64 num_perm)
{
	uniform unsigned int64 first, last, span;
	span = num_perm / taskCount;
	first = taskIndex * span;
	last = min((taskIndex+1) * span, num_perm);

	foreach (i = first ... last)
	{
		find_permutation(word, num_perm, i);
	}
}

static task void perm_t_task(uniform int word[], uniform unsigned int64 num_perm , uniform int num_threads)
{
	     uniform unsigned int64 first, last, span;
        span = num_perm/taskCount;
        first = taskIndex * span;
        last = min((taskIndex+1) * span, num_perm);

        foreach (i = first ... last)
        {
      		// in this scope:
      		// we must split work into partitions
      		// so total count s the span
      		// i is the start of the span for threads
      		unsigned int64 first_t, last_t, span_t;
      		span_t = span / num_threads;
      		first_t = i * span_t;
      		last_t = min((i+1) * span_t, span);
      		unsigned int64 i_t;
      		for (i_t = first_t; i_t < last_t; i_t++)
      		{
              find_permutation(word, num_perm, i_t);
      		}
        }
}

//-------------------The following are the functions that are called from main-------------------//

export void perm_ispc(uniform int word[], uniform unsigned int64 num_perm , uniform int num_threads)
{
	if (num_threads <= 1)
	{
		perm_no_task(word, num_perm);
	}
	else
	{
		perm_t_no_task(word, num_perm, num_threads);
	}
}

export void perm_ispc_tasks(uniform int word[], uniform unsigned int64 num_perm , uniform int num_threads, uniform int num_cores2)
{
	if (num_threads <= 1)
	{
		launch[num_cores2] perm_task(word, num_perm);
	}
	else
	{
		launch[num_cores2] perm_t_task(word, num_perm, num_threads);
	}
	sync;
}
