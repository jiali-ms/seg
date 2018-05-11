"""
The regex blocking method is from jieba.
https://github.com/fxsjy/jieba/blob/master/jieba/__init__.py

Algorithm wise, use dp with uni-gram probability.
"""

from collections import defaultdict, deque
import math
import time
import re

re_dict = re.compile('^(.+?)( [0-9]+)?( [a-z]+)?$', re.U)
re_han = re.compile("([\u4E00-\u9FD5]+)", re.U)
re_skip = re.compile("[^a-zA-Z0-9+#\n]", re.U)

class Lexicon():
    def __init__(self, dict_path):
        """
        Init lexicon with dict path.

        Format is 'word freq pos' with space separated. Note that we don't handle pos so far.
        :param dict_path:
        """
        self.total = 0

        self.dict = {}
        with open(dict_path, 'r', encoding='utf-8')as f:
            for line in f:
                word, freq, tag = re_dict.match(line).groups()
                if freq is not None:
                    freq = freq.strip()

                # give a minimal 1 count for rare words without freq as smoothing
                freq = max(int(freq), 1)
                self.dict[word] = freq
                self.total += freq

                # prefix but not yet a word will be 0
                # mimic of prefix check of trie for acceleration
                for i in range(len(word)):
                    sub_word = word[:i + 1]
                    if sub_word not in self.dict:
                        self.dict[sub_word] = 0

    def check_prob(self, word):
        """
        Return prob in neg log format.

        :param word:
        :return: 0 for prefix, neg log for word. Otherwise None
        """
        if word in self.dict:
            freq = self.dict[word]
            if freq is not 0:
                return -math.log(freq/self.total)
            else:
                return 0
        else:
            return None

    def has_prefix(self, word):
        return word in self.dict

    def is_word(self, word):
        return word in self.dict and self.dict[word] != 0

class Decoder():
    def __init__(self):
        # model will provide probability
        self.lexicon = Lexicon('user_dict.txt')

    def decode(self, input):
        """
        decode the input sentence.

        This method cut input sentence into blocks first with non-chinese symbols as natural boundary.
        It is vital for speed up. In local experiment, 50x faster.
        :param input:
        :return:
        """

        blocks = re_han.split(input)
        for block in blocks:
            if not block:
                continue
            if re_han.match(block):
                for word in self.decode_(block):
                    yield word
            else:
                if block == '':
                    continue
                else:
                    matched = False
                    tmp = re_skip.split(block)
                    for x in tmp:
                        if re_skip.match(x):
                            matched = True
                            yield x
                    if not matched:
                        yield block

    def decode_(self, input):
        """
        use dp to find best path.

        This method decode with backward lookup. Notice that forward lookup is also a choice.
        :param input: The raw input sequence
        :return: Best path as list of words
        """

        # build frames
        # frame is backward lookup with start_idx to key as valid word
        frames = defaultdict(list)
        input_size = len(input)
        for s in range(input_size):
            e = s + 1
            while self.lexicon.has_prefix(input[s:e]) and e <= input_size:
                if self.lexicon.is_word(input[s:e]):
                    frames[e].append(s)
                    e += 1

            # in case of oov symbols, segment to char
            if s not in frames:
                frames[s] = [(s-1, 0)]

        # decode best path with simple dp from start
        best_path = {}
        best_path[0] = (0, 0)
        for i in range(1, input_size + 1):
            for s in frames[i]:
                word = input[s:i]
                prob = self.lexicon.check_prob(word)
                neg_log = prob + best_path[s][1]
                if i not in best_path or neg_log < best_path[i][1]:
                    best_path[i] = (s, neg_log)

        # parse results
        result = deque()
        idx = input_size
        while idx > 0:
            s = best_path[idx][0]
            result.appendleft(input[s:idx])
            idx = s

        for x in result:
            yield x

if __name__ == "__main__":
    decoder = Decoder()
    start_time = time.time()
    result = decoder.decode('结婚的和尚未结婚的，都是很nice cool的“靠谱人士”')
    end_time = time.time()
    print(' '.join(result))
    print('{} s'.format(end_time - start_time))