# segmentation
It is a study log of the segmentation algorithms for Chinese and Japanese. 

Segmentation is a field with different aspects. From application point of view, narrator wants to find the best pause to match human hearing experience. An input method needs to find best separation for user typing habit. Search engine may want to find trending new words, etc. There is no one best solution, but depends on the scenarios. 

While there are open source tools for direct use, here we want to keep core algorithms simple in python for learning purpose. 

## Lexicon based algorithms
The most practical way for Chinese and Japanese is lexicon based methods. 
### Uni-gram dp
Uni-gram dp is a simplified version for LM decoder. You can imagine it as the 1 beam size decoder. It is the fastest way for Chinese decoding and the default algorithm for tools like *jieba*. The speed should be 1-20Mb/s. The probability of sentence is simplified as the 

$$p(w)=p(w_{1})p(w_{2})...p(w_{n})$$

See example [lexicon_dp.py](https://github.com/jiali-ms/seg/blob/master/lexicon_dp.py). It applied several key tricks from *jieba* for acceleration. The tested speed with intel E5 CPU is about 500K sentences per second. 

Use *jieba* 300K  [dict](https://github.com/fxsjy/jieba/blob/master/jieba/dict.txt) if you don't have one in hand to play.

### LM decoder
Japanese has many functional words that are hard to tell from single char or part of word, like 'の', it needs a language model for bi-gram or tri-gram probability and a decoder to find best path as segmentation results. 

TBD

## Sequence labeling based algorithms
TBD
