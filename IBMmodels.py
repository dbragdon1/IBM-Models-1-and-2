
def read_corpus_addnull(english_corpus, foreign_corpus):
    "Reads a corpus and adds in the NULL word."
    english = [["*"] + e_sent.split() for e_sent in open(english_corpus, encoding = 'utf-8')]
    foreign = [f_sent.split() for f_sent in open(foreign_corpus, encoding = 'utf-8')]
    return english, foreign

english_corpus = 'corpus.en'
spanish_corpus = 'corpus.es'
en_corpus, es_corpus = read_corpus_addnull(english_corpus, spanish_corpus)

class IBM_1():
	def __init__(self, es_corpus, en_corpus, S):
		self.n_e = {}
		self.es_corpus = es_corpus 
		self.en_corpus = en_corpus
		self.S = S #number of iterations
		self.t_probs = {} #dictionary of alignment probabilities
		self.wordpairs = set()
	
	def calc_n_e(self):
		for e_sent, f_sent in zip(self.en_corpus, self.es_corpus):
			for e in e_sent:
				for f in f_sent:
					wordpair = (e,f)
					if wordpair not in self.wordpairs:
						self.wordpairs.add(wordpair)
						if e not in self.n_e:
							self.n_e[e] = 1
						else: self.n_e[e] += 1
		return self.n_e

	def init_t_probs(self):
		for e, f in self.wordpairs:
			self.t_probs[(f,e)] = 1 / self.n_e[e]
		return self.t_probs

	def fit(self):
		self.n_e = self.calc_n_e()
		self.t_probs = self.init_t_probs()
		print('Estimating t(f|e) with %s iterations: \n' % self.S)

		for s in range(self.S):

			print("Iteration %s" % (s + 1))

			count_dict = {}
			count_e_dict = {}

			for f_sent, e_sent in zip(self.es_corpus, self.en_corpus):
				for f in f_sent:
					denom = sum([self.t_probs[(f,e)] for e in e_sent])

					for e in e_sent:
						d = self.t_probs[(f,e)] / denom
						
						if (e,f) not in count_dict:
							count_dict[(e,f)] = d 
						else: count_dict[(e,f)] += d

						if e not in count_e_dict:
							count_e_dict[e] = d 
						else: count_e_dict[e] += d
			for f, e in self.t_probs:
				self.t_probs[(f,e)] = count_dict[(e,f)] / count_e_dict[e]
		return self.t_probs

class IBM_2():
	def __init__(self, es_corpus, en_corpus, t_probs, S):
		self.es_corpus = es_corpus
		self.en_corpus = en_corpus 
		self.S = S
		self.t_probs = t_probs #initialized from Model 1
		self.q_dict = {}

	def init_q_dict(self):
		for f_sent, e_sent in zip(self.es_corpus, self.en_corpus):
			m = len(f_sent)
			l = len(e_sent)
			for i, f in enumerate(f_sent):
				for j, e in enumerate(e_sent):
					self.q_dict[(j, i + 1, l, m)] = 1 / l

		return self.q_dict

	def fit(self):
		
		print('Estimating q(j|i,l,m) and t(f|e)')

		self.q_dict = self.init_q_dict()

		for s in range(self.S):

			print("Iteration %s" % (s + 1))

			count_dict = {}
			count_e_dict = {}
			count_j_dict = {}
			count_i_dict = {}

			for f_sent, e_sent in zip(self.es_corpus, self.en_corpus):
				l = len(e_sent)
				m = len(f_sent)
				k = self.es_corpus.index(f_sent)

				for i, f in enumerate(f_sent):
					denom = sum([self.q_dict[(j, i + 1, l, m)] * self.t_probs[(f,e)] for j, e in enumerate(e_sent)])
					for j, e in enumerate(e_sent):
						num = self.q_dict[(j, i + 1, l, m)] * self.t_probs[(f,e)]
						d = num / denom

						if (e,f) not in count_dict:
							count_dict[(e,f)] = d
						else: count_dict[(e,f)] += d

						if e not in count_e_dict:
							count_e_dict[e] = d
						    
						else: count_e_dict[e] += d

						if (j, i + 1, l, m) not in count_j_dict:
							count_j_dict[(j, i + 1, l, m)] = d
						    
						else: count_j_dict[(j, i + 1, l, m)] += d

						if (i + 1, l, m) not in count_i_dict:
							count_i_dict[(i + 1, l, m)] = d
						    
						else: count_i_dict[(i + 1, l, m)] += d

			for j, i, l, m in self.q_dict:
				self.q_dict[(j, i, l, m)] = count_j_dict[(j, i, l, m)] / count_i_dict[(i, l, m)]
		    
			for f, e in self.t_probs:
				self.t_probs[(f,e)] = count_dict[(e,f)] / count_e_dict[e]

		return self.q_dict, self.t_probs


if __name__ == '__main__':

	print('Fitting corpus to IBM Model 1.\n')

	model1 = IBM_1(es_corpus, en_corpus, S = 5)

	alignments = model1.fit()

	english_dev = 'dev.en'
	spanish_dev = 'dev.es'

	en_dev, es_dev = read_corpus_addnull(english_dev, spanish_dev)

	print('\nFinished Training Model 1. Writing Predictions to file.\n')

	preds = open('dev.p1.out', 'w')
	for e_sent, f_sent in zip(en_dev, es_dev):
		for i, f in enumerate(f_sent):
			aligns = [alignments[(f,e)] for e in e_sent[1:]]
			max_idx = aligns.index(max(aligns))
			sent_idx = str(es_dev.index(f_sent) + 1)
			preds.write(sent_idx + ' ' + str(max_idx + 1) + ' ' + str(i + 1) + '\n')

	preds.close()



	print('Fitting corpus and alignments to IBM Model 2.\n')

	model2 = IBM_2(es_corpus, en_corpus, alignments, S = 5)




	q_dict, alignment_probs = model2.fit()

	print('Finished Training Model 2. Writing Predictions to file.\n')

	preds2 = open('dev.p2.out', 'w')

	for e_sent, f_sent in zip(en_dev, es_dev):
		for i, f in enumerate(f_sent):
			l = len(e_sent)
			m = len(f_sent)
			aligns = [q_dict[(j + 1, i + 1, l, m)] * alignment_probs[(f,e)] for j, e in enumerate(e_sent[1:])]
			max_idx = aligns.index(max(aligns))
			sent_idx = str(es_dev.index(f_sent) + 1)
			preds2.write(sent_idx + ' ' + str(max_idx + 1) + ' ' + str(i + 1) + '\n')

	preds2.close()


