import os
import re
import pickle

util_path = "" #specify path to LIWC dictionaries""

class LIWCFactory(object):

    def __init__(self, transcripts=[], lang='eng'):
        """Processes the given list of lists containing string tokens named transcripts 
        to produce a list of LIWC feature vectors.
        """
        self.lang = lang
        liwc_dict = self.buildLIWCDict(self.lang)
        self.idmap = liwc_dict["idmap"]
        self.wordmap = liwc_dict["wordmap"]
        self.categories = liwc_dict["categories"]
        self.dimensions = len(self.categories.keys())
        self.transcripts = transcripts
        self.vectors = map(self.computeLIWC, transcripts)
 
    def setTranscripts(self, transcripts):
        """Defines the tokenized transcripts to process"""
        self.transcripts = transcripts
        self.vectors = map(self.computeLIWC, transcripts)        
        return self
 
    def getLIWCVectors(self):
        """Get the list of LIWC feature vectors corresponding to the tokenized transcripts"""
        return self.vectors

    def computeLIWC(self, words):
        """Compute the LIWC vector for a list of words (string tokens)"""
        nonchars = re.compile(r'[^a-z\'-@]')
        # initialize the LIWC vector
        liwc_vector = [0] * self.dimensions

        # for each word in the tokenized list
        for word in words:
            word = word.lower()
            word = nonchars.sub("", word)
            # if the word is in the LIWC dictionary, find which categories need to be updated 
            # according to LIWC, and increment the corresponding entry in the vector
            if word in self.wordmap.keys():
                categories = [self.idmap[idx] for idx in self.wordmap[word]]
                liwc_vector = [x + 1 if idx in categories else x for (idx, x) in enumerate(liwc_vector)]
                continue
            word_length = len(word)
        
            # if the word is not in the LIWC dictionary, use a greedy approach to select the 
            # best match. e.g: 'christianity' is not in the LIWC dictionary, but 'christian*' is
            for index in range(0, word_length):
                if word[0:word_length-index] + '*' in self.wordmap.keys():
                    categories = [self.idmap[idx] for idx in self.wordmap[word[0:word_length-index] + '*']]
                    liwc_vector = [x + 1 if idx in categories else x for (idx, x) in enumerate(liwc_vector)]
                    continue
        
            # if the greedy approach yields no match, skip the word altogether. 
            # modify to account for 'unknown' words?
        return [float(i)/len(words) for i in liwc_vector]

        # compute coarse_featurs: number of words, number of unique words, # of six letter words
#        coarse_feats = [len(words), len(set(words)), sum([1 for i in words if len(i) >= 6]) * 1.0/len(words)]

        # normalize LIWC feature vector and append it to the coarse feature vector

#        coarse_feats.extend([i*1.0/len(words) for i in liwc_vector])
#        return coarse_feats
                               

#        coarse_feats.extend([float(i)/len(words) for i in liwc_vector])
#        return coarse_feats               

    def buildLIWCDict(self, lang):
        """Build the LIWC dictionary from scratch"""
        
        # if the LIWC dictionary has already been generated and pickled, return it!
        if os.path.exists(os.path.join(util_path, "liwcdict_%s.pkl" % lang)):
            infile = open(os.path.join(util_path, "liwcdict_%s.pkl" % lang), "rb")
            liwc_dict = pickle.load(infile)
            return liwc_dict

        if lang == 'eng':
            liwc_ref = "LIWC2007.txt"
        elif lang == 'span':
            liwc_ref = "LIWC2001_Spanish.dic"
        # load the dictionary file provided by LIWC
        liwc_file = open(os.path.join(util_path, "%s" % liwc_ref), "r")
        liwc_pkl = open(os.path.join(util_path, "liwcdict_%s.pkl" % lang), "wb")
        liwc_dict = {}

        # read the contents of the dictionary file
        stream = liwc_file.read()
        lines = stream.splitlines()[1:]
        
        # the header contains a mapping from function word class to a unique id. 
        # this section is demarcated by '%' in the file
        hdr_idx = lines.index('%')
        header = lines[0:hdr_idx]
        
        # initialize 3 dictionaries: function word class name to LIWC id, LIWC id to a 
        # fixed/sequential order id (0-64), and a mapping from a word in the LIWC dictionary 
        # to a list of category id's that correspond to the usage of that word
        catToid = {}
        idToidx = {}
        wordmap = {}
        
        # build the first dictionary from the header, and the second dictionary by sequential 
        # ordering
        for (idx, line) in enumerate(header):
            (cat_id, cat) = line.split('\t')
            catToid[cat] = cat_id
            idToidx[cat_id] = idx
        
        # build the third dictionary by setting each word in the dictionary as a key and the 
        # list of category id's as the value
        for (idx, line) in enumerate(lines[hdr_idx + 1:]):
            mapping = line.split('\t')
            k,v = mapping[0], mapping[1:]
            wordmap[k] = v
        
        # package dictionaries into one primary dictionary
        liwc_dict["categories"] = catToid
        liwc_dict["idmap"] = idToidx
        liwc_dict["wordmap"] = wordmap
        
        # pickle primary dictionary for quick subsequent access 
        pickle.dump(liwc_dict, liwc_pkl)
        liwc_pkl.close()
        return liwc_dict
