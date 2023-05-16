# A data processor for RNN project


import os 


class DataHandler:
    def __init__(self):
        self.txt_path = None
        self.sentences = []
        self.chars = None
        self.intChar = None
        self.charInt = None
        self.vocab_size = 0


    def process_txt(self, txt_file):
        """
        Read in a text file. Save all lines that are not a newline
        
        :param txt_file: a text file
        """
        with open(txt_file) as f:
            lines = f.readlines()
            lines = [line.strip('\n') for line in lines if line != '\n']
            self.sentences = lines

    
    def get_chars(self, sentences):
        """
        Get all the unique characters that are in the provided sentences
        and set the self.chars varaiable to be this set

        :param sentences: a list of strings
        """
        self.chars = set(''.join(sentences))

    
    def set_vocab(self, chararacters):
        """
        Create 2 dicionaries that are the inverse of eachother
        Label each character with a number (key, value) and vis versa

        :param characters: a set of characters
        """
        self.intChar = dict(enumerate(chararacters))
        self.charInt = {character: index for index, character in self.intChar.items()}
        self.set_vocab_size(self.intChar)
        self.intChar[self.vocab_size] = '-PAD-'
        self.charInt['-PAD-'] = self.vocab_size


    def set_vocab_size(self, vocab):
        """
        Record the number of unique characters seen

        :param vocab: a dictionary of characters as keys and an integer as a value
        """
        self.vocab_size = len(vocab)


def main():
    dh = DataHandler()
    dh.process_txt(os.path.join(os.path.dirname(__file__), 'tiny-shakespeare.txt'))
    dh.get_chars(dh.sentences)
    dh.set_vocab(dh.chars)
    dh.set_vocab_size(dh.charInt)
    print(dh.sentences)
    print(dh.chars)
    print(dh.intChar)
    print(dh.charInt)



if __name__ == '__main__':
    main()