"""
Information Theory Lab

Sam Goldrup
Section 001
27 September 2022
"""


import numpy as np
import wordle

# Problem 1
def get_guess_result(true_word, guess):
    """
    Returns an array containing the result of a guess, with the return values as follows:
        2 - correct location of the letter
        1 - incorrect location but present in word
        0 - not present in word
    For example, if the true word is "boxed" and the provided guess is "excel", the 
    function should return [0,1,0,2,0].
    
    Arguments:
        true_word (string) - the secret word
        guess (string) - the guess being made
    Returns:
        result (array of integers) - the result of the guess, as described above
    """
    lets,w,g = list(true_word),np.array(list(true_word)),np.array(list(guess)) #list of letters, extra list for tracking
    scores = 2*(w==g)
    idxs = list(np.where(scores>0))[0] #indices where scores are nonzero
    idxs_0 = list(np.where(scores==0))[0] #....zero
    for idx in idxs:
        lets[idx] = '0' #remove the letter from lets without having to keep track of indices

    for idx in idxs_0: #iterate over the indices where scores are zero
        let = g[idx] #save the letter
        if let in w and let in lets:
            let_dex = lets.index(let) #find index in lets where let lives
            lets[let_dex] = '0' #"remove" it
            scores[idx] += 1 #make it grey

    return scores

# Problem 2
def load_words(filen):
    """
    Loads all of the words from the given file, ensuring that they 
    are formatted correctly.
    """
    with open(filen, 'r') as file:
        # Get all 5-letter words
        words = [line.strip() for line in file.readlines() if len(line.strip()) == 5]
    return words
    
def get_all_guess_results(possible_words, allowed_words):
    """
    Calculates the result of making every guess for every possible secret word
    
    Arguments:
        possible_words (list of strings)
            A list of all possible secret words
        allowed_words (list of strings)
            A list of all allowed guesses
    Returns:
        ((n,m,5) ndarray) - the results of each guess for each secret word,
            where n is the the number
            of allowed guesses and m is number of possible secret words.
    """
    #each row is results of i-th guess on every secret word
    #each column is results of every guess on j-th secret word
    results = np.array([[get_guess_result(true_word,guess) for true_word in possible_words] for guess in allowed_words])
    np.save("results", results)
    
# Problem 3
def compute_highest_entropy(all_guess_results, allowed_words):
    """
    Compute the entropy of each guess.
    
    Arguments:
        all_guess_results ((n,m,5) ndarray) - the output of the function
            from Problem 2, containing the results of each 
            guess for each secret word, where n is the the number
            of allowed guesses and m is number of possible secret words.
        allowed_words (list of strings) - list of the allowed guesses
    Returns:
        (string) The highest-entropy guess
        (int) Index of the highest-entropy guess
    """
    #reduce dimensionality
    def ternary(a,b,c,d,e):
        return a*1 + b*3 +c*9 + d*27 + e*81

    #call function on all five 'columns'
    terned = ternary(all_guess_results[:,:,0],all_guess_results[:,:,1],all_guess_results[:,:,2],all_guess_results[:,:,3],all_guess_results[:,:,4])

    entropies = []
    for ent in terned:
        _, counts = np.unique(ent,return_counts=True)
        entropy = np.sum([num/len(ent) * -np.log2(num/len(ent)) for num in counts]) #entropy calculation
        entropies.append(entropy)

    max_ind = np.argmax(entropies) #'soare' should be the best candidate

    return allowed_words[max_ind], max_ind

# Problem 4
def filter_words(all_guess_results, possible_words, guess_idx, result):
    """
    Create a function that filters the list of possible words after making a guess.
    Since we already computed the result of all guesses for all possible words in 
    Problem 2, we will use this array instead of recomputing the results.
    
	Return a filtered list of possible words that are still possible after 
    knowing the result of a guess. Also return a filtered version of the array
    of all guess results that only contains the results for the secret words 
    still possible after making the guess. This array will be used to compute 
    the entropies for making the next guess.
    
    Arguments:
        all_guess_results (3-D ndarray)
            The output of Problem 2, containing the result of making
            any allowed guess for any possible secret word
        possible_words (list of str)
            The list of possible secret words
        guess_idx (int)
            The index of the guess that was made in the list of allowed guesses.
        result (tuple of int)
            The result of the guess
    Returns:
        (list of str) The filtered list of possible secret words
        (3-D ndarray) The filtered array of guess results
    """
    #filter is np.all(all_guess_results[guess_idx] == result,axis=1)
    filtered_results = all_guess_results[:,np.all(all_guess_results[guess_idx] == result,axis=1),:] #filter along axis 1
    filtered_secret = np.array(possible_words)[np.all(all_guess_results[guess_idx] == result,axis=1)] #filtered secret words

    return filtered_secret,filtered_results

# Problem 5
def play_game_naive(game, all_guess_results, possible_words, allowed_words, word=None, display=False):
    """
    Plays a game of Wordle using the strategy of making guesses at random.
    
    Return how many guesses were used.
    
    Arguments:
        game (wordle.WordleGame)
            the Wordle game object
        all_guess_results ((n,m,5) ndarray)
            an array as outputted by problem 2 containing the results of every guess for every secret word.
        possible_words (list of str)
            list of possible secret words
        allowed_words (list of str)
            list of allowed guesses
        
        word (optional)
            If not None, this is the secret word; can be used for testing. 
        display (bool)
            If true, output will be printed to the terminal by the game.
    Returns:
        (int) Number of guesses made
    """
    # Initialize the game
    game.start_game(word=word, display=display)
    real_word = game.word #get the word

    num_guesses = 0

    while len(possible_words) > 1:
        i = np.random.randint(0,len(allowed_words)) #random guess gen
        new_guess = allowed_words[i]
        res,num_guesses = game.make_guess(new_guess)

        if real_word == new_guess: #solved!
            break
        possible_words,all_guess_results = filter_words(all_guess_results,possible_words,i,res) #update prior
    
    res, num_guesses = game.make_guess(possible_words[0])
    return num_guesses

# Problem 6
def play_game_entropy(game, all_guess_results, possible_words, allowed_words, word=None, display=False):
    """
    Plays a game of Wordle using the strategy of guessing the maximum-entropy guess.
    
    Return how many guesses were used.
    
    Arguments:
        game (wordle.WordleGame)
            the Wordle game object
        all_guess_results ((n,m,5) ndarray)
            an array as outputted by problem 2 containing the results of every guess for every secret word.
        possible_words (list of str)
            list of possible secret words
        allowed_words (list of str)
            list of allowed guesses
        
        word (optional)
            If not None, this is the secret word; can be used for testing. 
        display (bool)
            If true, output will be printed to the terminal by the game.
    Returns:
        (int) Number of guesses made
    """
    # Initialize the game
    game.start_game(word=word, display=display)
    real_word = game.word

    num_guesses = 0

    while len(possible_words) > 1:
        new_guess,i = compute_highest_entropy(all_guess_results, allowed_words) #non-random, data-driven guess
        res,num_guesses = game.make_guess(new_guess)

        if real_word == new_guess:
            break
        possible_words,all_guess_results = filter_words(all_guess_results,possible_words,i,res) #update prior
    
    res, num_guesses = game.make_guess(possible_words[0])
    return num_guesses

# Problem 7
def compare_algorithms(all_guess_results, possible_words, allowed_words, n=20):
    """
    Compare the algorithms created in Problems 5 and 6. Play n games with each
    algorithm. Return the mean number of guesses the algorithms from
    problems 5 and 6 needed to guess the secret word, in that order.
    
    
    Arguments:
        all_guess_results ((n,m,5) ndarray)
            an array as outputted by problem 2 containing the results of every guess for every secret word.
        possible_words (list of str)
            list of possible secret words
        allowed_words (list of str)
            list of allowed guesses
        n (int)
            Number of games to run
    Returns:
        (float) - average number of guesses needed by naive algorithm
        (float) - average number of guesses needed by entropy algorithm
    """
    naive_guesses = []
    entropy_guesses = []
    for i in range(n):
        game = wordle.WordleGame() #initialize new game
        naive_guesses.append(play_game_naive(game, all_guess_results, possible_words, allowed_words, word=None, display=False))
        entropy_guesses.append(play_game_entropy(game, all_guess_results, possible_words, allowed_words, word=None, display=False))

    #compare the means
    return np.mean(naive_guesses), np.mean(entropy_guesses)