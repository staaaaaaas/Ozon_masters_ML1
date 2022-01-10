def check_first_sentence_is_second(str1, str2):
    word_list1 = str1.split()
    word_list2 = str2.split()
    word_dict1 = {}
    for word in word_list1:
        word_dict1[word] = word_dict1.get(word, 0) + 1
    for word in word_list2:
        if word_dict1.get(word, 0) == 0:
            return False
        else:
            word_dict1[word] -= 1
    return True
