import re


# class SimpleTokenizerV2:
#     def __init__(self, vocab):
#         self.str_to_int = vocab
#         self.int_to_str = {index: token for token, index in vocab.items()}

#     def encode(self, text):
#         preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text) # seperate words based on space & grammer
#         preprocessed = [item.strip() for item in preprocessed if item.strip()]
#         preprocessed = [
#             item if item in self.str_to_int
#             else '<|unk|>' for item in preprocessed
#         ]
#         return [self.str_to_int[item] for item in preprocessed]

#     def decode(self, ids):
#         text = ' '.join(self.int_to_str[index] for index in ids)
#         return re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)

# Now, the SimpleTokenizerV3 keeps the whitespace tokens so that decoding can restore the original text

class SimpleTokenizerV3:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {index: token for token, index in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item for item in preprocessed if item != '']
        return [self.str_to_int.get(item, self.str_to_int['<|unk|>']) for item in preprocessed]

    def decode(self, ids):
        return ''.join(self.int_to_str[index] for index in ids)