import os
import re
from operator import itemgetter    
 
file_dir = "C:/Users/Fan_2019/Downloads/Parallel_Project_paper/Get the document collection and E(entries) by python/washington"
file_dict={} # dict of {filename: file number}
most_freq_word_file_dict = {} # dict of {filename: [most_freq_word, freq]}
all_words_each_file_dict = {} # dict of {filename: {all words in the file : freq}}
all_most_freq_words = [] # list of all most freq words from all files

f_num = 1
for each_file in os.listdir(file_dir):
    file_dict[each_file] = f_num
    f_num = f_num + 1

    frequency = {}
    full_file_name = file_dir + '/' + each_file
    open_file = open(full_file_name, 'r')
    file_to_string = open_file.read()
    words = re.findall(r'(\b[A-Za-z][a-z]{2,9}\b)', file_to_string)

    for word in words:
        count = frequency.get(word,0)
        frequency[word] = count + 1
    
    sorted_freq = {}
    for key, value in reversed(sorted(frequency.items(), key = itemgetter(1))):
        sorted_freq[key] = value
    
    # get the most freq word from dict of {word:freq}
    most_freq_word_current_file = list(sorted_freq.keys())[0]

    # make a list of most freq word and its freq
    word_freq = sorted_freq[most_freq_word_current_file]
    most_freq_word_list = [most_freq_word_current_file, word_freq]
   
    # add word list to dict
    most_freq_word_file_dict[each_file] = most_freq_word_list
    all_words_each_file_dict[each_file] = sorted_freq
    all_most_freq_words.append(most_freq_word_current_file)

all_most_freq_words_unique = sorted(list(set(all_most_freq_words))) # to eliminate duplicates


# assign no. to all most freq words {word,number}
all_most_freq_words_num = {}
num = 1
for w in all_most_freq_words_unique:
    all_most_freq_words_num[w] = num
    num = num + 1

#print(all_most_freq_words_num)
#print(file_dict)

# how many most freq words each file has
# this is the output dictonary we want
each_file_freq_words_dict = {}
for file_name in most_freq_word_file_dict.keys():
    
    print("File num : ", file_dict[file_name])

    #get the dict of {all words in the file : freq} in current file
    a = all_words_each_file_dict[file_name]

    # get current freq word, its freq and its number
    b = most_freq_word_file_dict[file_name]
    freq_word = b[0]
    freq = b[1]

    freq_word_num = all_most_freq_words_num[freq_word]
    #print("Most freq word : ", freq_word_num)

    # create a dict to keep all freq words and respective numbers in current file 
    # {word : [number, freq]}
    current_freq_word_num_dict = {}
    current_freq_word_num_dict[freq_word] = [freq_word_num, freq]
    #print("\nAdding most freq word for current file : ", freq_word_num,"\n")

    # now check other freq words, if they exists in current file
    #print("\nAdding other freq words:\n")
    for most_freq_word in all_most_freq_words_unique:
        if((most_freq_word != freq_word) and (most_freq_word in list(a.keys()))):
            current_num = all_most_freq_words_num[most_freq_word]
            #print("word num : ", current_num)
            current_freq = a[most_freq_word]
            #print("freq : ", current_freq)
            current_freq_word_num_dict[most_freq_word] = [current_num, current_freq]

    # get file num
    file_num = file_dict[file_name]

    # assign {file number : [word number, freq]}
    each_file_freq_words_dict[file_num] = current_freq_word_num_dict.values()
    #print("File words: ", file_num ,each_file_freq_words_dict[file_num], "\n")

for f_num in list(each_file_freq_words_dict.keys()):
    wrd_num_freq = each_file_freq_words_dict[f_num]
    wrd_num_freq_list = list(wrd_num_freq)
    #print(wrd_num_freq_list)
    
    # store all freq of current file in one list
    freq_list = []
    for j in wrd_num_freq_list:
        # [number, freq], separate freq
        freq_list.append(j[1])

    # calculate the sum of all freq
    sum_freq = sum(freq_list)

    # calculate the norm for each word number
    for w_frq in wrd_num_freq_list:
        norm = w_frq[1]/(sum_freq - w_frq[1])
        w_frq[1] = norm

    #print(wrd_num_freq_list)

    # assign {file number : [word number, freq]}
    each_file_freq_words_dict[f_num] = wrd_num_freq_list


#print(each_file_freq_words_dict)

result = [[],[],[]]
for f_num, wrds_freq_list in list(each_file_freq_words_dict.items()):
    # repeat file no. that many times
    for wrd_frq in list(wrds_freq_list):
        result[0].append(f_num)
        result[1].append(wrd_frq[0])
        result[2].append(wrd_frq[1])

f = open("C:/Users/Fan_2019/Downloads/result_norm.txt", "w")
f.write('%s' % result)















