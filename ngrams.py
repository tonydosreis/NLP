import numpy as np

def token_to_word(token, tokenizer):
    return tokenizer.sequences_to_texts([[token + 1]])[0]

def tokens_to_words(tokens, tokenizer):
    for token in tokens:
        print(token_to_word(token,tokenizer), end = " ")
    print()

def get_ngram_freq(data, n = 1):
    
    #data -> sequence of tokens
    #n -> order of n-gram
    
    new_size = data.shape[0] - n + 1
    
    assert n >= 1
    temp = np.zeros( (new_size, n), dtype = int )
    
    i_init = 0
    i_fin = new_size
    
    #filling each column
    for i in range(n):
        temp[:,i] = data[i_init:i_fin]
        i_init += 1
        i_fin += 1
                
    unique_ngrams, counts = np.unique(temp, axis = 0, return_counts = True)
    freq = counts/counts.sum()
    
    return unique_ngrams, freq

def sample_prob(prob):
    cum_prob = np.cumsum(prob)
    cum_prob /= cum_prob.max() # just in case max value is diff than 1
    
    i = np.random.random()
    return np.argmax(cum_prob > i)

def sample_ngram(unique_ngrams, freq):
    i = sample_prob(freq)
    return unique_ngrams[i,:]

def sample_ngram_cond(unique_ngrams, freq, cond):    
        
    n = unique_ngrams.shape[1]
    
    temp = np.all(unique_ngrams[:,:n-1] == cond, axis = 1)

    unique_ngram_cond = unique_ngrams[temp]
    freq_cond = freq[temp]
    freq_cond /= freq_cond.sum()
    
    return sample_ngram(unique_ngram_cond,freq_cond)[-1]

def generate_text(original_text, n, text_len):
    
    assert type(n) == int
    assert n > 0
    assert text_len > n
    
    text_seq = np.zeros((text_len), dtype = int)
    
    
    for i in range(n - 1):
        if(i == 0):
            unique_ngrams, freq = get_ngram_freq(original_text, n = 1)
            token = sample_ngram(unique_ngrams, freq)
        else:
            unique_ngrams, freq = get_ngram_freq(original_text, n = i + 1)
            token = sample_ngram_cond(unique_ngrams, freq, text_seq[:i])
            
        text_seq[i] = token
        
    next_i = n - 1
            
    unique_ngrams, freq = get_ngram_freq(original_text, n = n)
    
    for i in range(next_i, text_len):
                
        token = sample_ngram_cond(unique_ngrams, freq, text_seq[i-n + 1:i])
        text_seq[i] = token
                
    for token in text_seq:
        print(token_to_word(token,tokenizer), end = " ")
    print()


if __name__ == "__main__":

    #tensorflow only used to download sample data
    from tensorflow import keras
    import random

    shakespeare_url = "https://homl.info/shakespeare" # shortcut URL
    filepath = keras.utils.get_file("shakespeare.txt", shakespeare_url)
    with open(filepath) as f:
        shakespeare_text = f.read()
        
    tokenizer = keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts([shakespeare_text])
    data = np.array(tokenizer.texts_to_sequences([shakespeare_text])).flatten() - 1

    generate_text(data, 5, 10)