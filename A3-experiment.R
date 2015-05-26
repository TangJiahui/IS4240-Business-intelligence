# Name: Tang Jiahui
# Matric No: A0119415J

#==========================================

library("plyr")
library("tm")
library("class")
library("RTextTools")
library("parallel")

categories = dir("20_newsgroup/")

# create bi-gram tokenizer
BigramTokenizer <- function(x) unlist(lapply(ngrams(words(x), 2), paste, collapse = " "), use.names = FALSE)

# =========================================
# Step 0: Simple Classifier Provided by RTextTools

# Anyway Machine Learning is all about parameters.
train.vol = 100
test.vol = 100
inter.sparse.const = 0.95
final.sparse.const = 0.998

sample.vol = train.vol + test.vol

combine.train.targets = vector()
combine.test.targets = vector()
combine.train.tdm = TermDocumentMatrix(Corpus(VectorSource(vector())))  #create an empty TDM
combine.test.tdm = TermDocumentMatrix(Corpus(VectorSource(vector())))

for (i in 1:length(categories)) {
  category = categories[i]
  print(category)
  assign(paste0(category, ".cor"), Corpus(DirSource(directory=paste0("20_newsgroup/", category))))   # alt.atheism.cor = Corpus(DirSource(directory=paste0("20_newsgroup/", category)))
  sample.ids = sample(1:length(get(paste0(category, ".cor"))), sample.vol)   # sample.ids = sample(1:length(alt.atheism.cor), sample.vol)
  train.ids = sample.ids[1:train.vol]
  test.ids = sample.ids[(train.vol+1):sample.vol]
  
  # alt.atheism.train.tdm = TermDocumentMatrix(alt.atheism.cor[train.ids], control=...)
  assign(paste0(category, ".train.tdm"), TermDocumentMatrix(get(paste0(category, ".cor"))[train.ids], control=list(removePunctunation=TRUE, toLower=TRUE, removeNumbers=TRUE, stemDocument=TRUE, stopwords("english"))))
  # Bigram: assign(paste0(category, ".train.tdm"), TermDocumentMatrix(get(paste0(category, ".cor"))[train.ids], control=list(tokenize = BigramTokenizer, removePunctunation=TRUE, toLower=TRUE, removeNumbers=TRUE, stemDocument=TRUE, stopwords("english"))))
  
  # alt.atheism.test.tdm = TermDocumentMatrix(alt.atheism.cor[test.ids], control=...)
  assign(paste0(category, ".test.tdm"), TermDocumentMatrix(get(paste0(category, ".cor"))[test.ids], control=list(removePunctunation=TRUE, toLower=TRUE, removeNumbers=TRUE, stemDocument=TRUE, stopwords("english"))))
  # Bigram: assign(paste0(category, ".test.tdm"), TermDocumentMatrix(get(paste0(category, ".cor"))[test.ids], control=list(tokenize = BigramTokenizer, removePunctunation=TRUE, toLower=TRUE, removeNumbers=TRUE, stemDocument=TRUE, stopwords("english"))))
  
  # alt.atheism.train.tdm = removeSparseTerms(alt.atheism.train.tdm, inter.sparse.const)
  assign(paste0(category, ".train.tdm"), removeSparseTerms(get(paste0(category,".train.tdm")), inter.sparse.const))
  
  # alt.atheism.test.tdm = removeSparseTerms(alt.atheism.test.tdm, inter.sparse.const)
  assign(paste0(category, ".test.tdm"), removeSparseTerms(get(paste0(category,".test.tdm")), inter.sparse.const))
  
  combine.train.tdm = c(combine.train.tdm, get(paste0(category, ".train.tdm")))  # combine.train.tdm = c(combine.train.tdm, alt.atheism.train.tdm)
  combine.test.tdm = c(combine.test.tdm, get(paste0(category, ".test.tdm")))  # combine.test.tdm = c(combine.test.tdm, alt.atheism.test.tdm)
  combine.train.targets = c(combine.train.targets, rep(i, train.vol))
  combine.test.targets = c(combine.test.targets, rep(i, test.vol))
}

combine.tdm = as.DocumentTermMatrix(c(combine.train.tdm, combine.test.tdm))  # Have to manually do this, it will not automatically transpose when creating container
combine.tdm = removeSparseTerms(combine.tdm, final.sparse.const)
combine.targets = c(combine.train.targets, combine.test.targets)

container = create_container(combine.tdm, combine.targets, trainSize=1:(length(categories)*train.vol), testSize=(length(categories)*train.vol+1):length(combine.targets), virgin=FALSE)


# Testing between models (based on final.sparse.const set to 0.95 for faster performance)

algorithms = print_algorithms()

do.model = function (algorithm) {
  print(algorithm)
  assign(paste0("model_", algorithm), train_model(container, algorithm), envir = .GlobalEnv)  # model_RF = train_model(container, "RF")
  print(paste("model done", algorithm))
  assign(paste0("classify_", algorithm), classify_model(container, get(paste0("model_", algorithm))), envir = .GlobalEnv)  # classify_RF = classify_model(container, model_RF)
  print(paste("classify done", algorithm))
  assign(paste0("analytics_", algorithm), create_analytics(container, get(paste0("classify_", algorithm))), envir = .GlobalEnv)  # analytics_RF = classify_model(container, classify_RF)
  print(paste("analytics done", algorithm))
  #assign(paste0("cross_", algorithm), cross_validate(container, 4, algorithm), envir = .GlobalEnv)  # cross_RF = cross_validate(container, 4, "RF")
  return(get(paste0("analytics_", algorithm)))
}

mclapply(algorithms, do.model, mc.cores=10)  # For run on SoC Clusters



# Highest is RF, let's test it (with final.sparse.const set to 0.998).
naive.model = train_model(container, "RF")

# Save the test result of naive classifier
naive.result = classify_model(container, naive.model)[,1]
naive.correct.result = as.factor(combine.test.targets)
naive.correct.p = sum(naive.result == naive.correct.result) / length(naive.correct.result)

# 0.9595

# ======================================
# Multi-step classifier
# Step 1 (Evaluation of First Level Classifiers & Construction of First Level Classifier)

first.level.cats = c("comp", "rec", "sci", "talk", "others")
train.vol = 300
test.vol = 300
inter.sparse.const = 0.95
final.sparse.const = 0.998

sample.vol = train.vol + test.vol

combine.first.train.targets = vector()
combine.first.test.targets = vector()
combine.second.test.targets = vector()
combine.train.tdm = TermDocumentMatrix(Corpus(VectorSource(vector())))
combine.test.tdm = TermDocumentMatrix(Corpus(VectorSource(vector())))

for (cat in first.level.cats) {
  assign(paste0(cat, ".train.tdm"), TermDocumentMatrix(Corpus(VectorSource(vector()))))  # comp.train.tdm = TermDocumentMatrix(Corpus(VectorSource(vector())))
  assign(paste0(cat, ".train.targets"), vector())  # comp.train.targets = vector()
}

for (i in 1:length(categories)) {
  category = categories[i]
  first.level.id = match(c(unlist(strsplit(category, split="[.]"))[1]), first.level.cats, nomatch=length(first.level.cats))
  print(category)
  
  assign(paste0(category, ".cor"), Corpus(DirSource(directory=paste0("20_newsgroup/", category))))  # comp.graphics.cor = Corpus(...)
  sample.ids = sample(1:length(get(paste0(category, ".cor"))), sample.vol)  # sample.ids = sample(1:length(comp.graphics.cor), sample.vol)
  train.ids = sample.ids[1:train.vol]
  test.ids = sample.ids[(train.vol+1):sample.vol]
  
  # comp.graphics.train.tdm = TermDocumentMatrix(comp.graphics.cor[train.ids], control=...)
  assign(paste0(category, ".train.tdm"), TermDocumentMatrix(get(paste0(category, ".cor"))[train.ids], control=list(removePunctunation=TRUE, toLower=TRUE, removeNumbers=TRUE, stemDocument=TRUE, stopwords("english"))))
  # comp.graphics.test.tdm = TermDocumentMatrix(comp.graphics.cor[test.ids], control=...)
  assign(paste0(category, ".test.tdm"), TermDocumentMatrix(get(paste0(category, ".cor"))[test.ids], control=list(removePunctunation=TRUE, toLower=TRUE, removeNumbers=TRUE, stemDocument=TRUE, stopwords("english"))))
  # comp.graphics.train.tdm = removeSparseTerms(comp.graphics.train.tdm, inter.sparse.const)
  assign(paste0(category, ".train.tdm"), removeSparseTerms(get(paste0(category,".train.tdm")), inter.sparse.const))
  # comp.graphics.test.tdm = removeSparseTerms(comp.graphics.test.tdm, inter.sparse.const)
  assign(paste0(category, ".test.tdm"), removeSparseTerms(get(paste0(category,".test.tdm")), inter.sparse.const))
  
  combine.train.tdm = c(combine.train.tdm, get(paste0(category, ".train.tdm")))  # combine.train.tdm = c(combine.train.tdm, comp.graphics.train.tdm)
  combine.test.tdm = c(combine.test.tdm, get(paste0(category, ".test.tdm")))  # combine.test.tdm = c(combine.test.tdm, comp.graphics.test.tdm)
  combine.first.train.targets = c(combine.first.train.targets, rep(first.level.id, train.vol))
  combine.first.test.targets = c(combine.first.test.targets, rep(first.level.id, test.vol))
  
  # comp.train.tdm = c(comp.train.tdm, comp.graphics.train.tdm)
  assign(paste0(first.level.cats[first.level.id], ".train.tdm"), c(get(paste0(first.level.cats[first.level.id], ".train.tdm")), get(paste0(category, ".test.tdm"))))
  # comp.train.targets = c(comp.train.targets, rep(i, test.vol))
  assign(paste0(first.level.cats[first.level.id], ".train.targets"), c(get(paste0(first.level.cats[first.level.id], ".train.targets")), rep(i, test.vol)))
  combine.second.test.targets = c(combine.second.test.targets, rep(i, test.vol))
}

combine.tdm = as.DocumentTermMatrix(c(combine.train.tdm, combine.test.tdm))
combine.tdm = removeSparseTerms(combine.tdm, final.sparse.const)
combine.first.targets = c(combine.first.train.targets, combine.first.test.targets)

container = create_container(combine.tdm, combine.first.targets, trainSize=1:(length(categories)*train.vol), testSize=(length(categories)*train.vol+1):length(combine.first.targets), virgin=FALSE)

# Testing between models

algorithms = print_algorithms()

for (algorithm in algorithms) {
  print(algorithm)
  cross_validate(container, 4, algorithm)
}


# ========== Result ==========
# This is based on final.sparse.const set to 0.95, for demo only.
# BAGGING: 0.8910506, 0.8494405, 0.9123173, 0.8962173 : 0.8872564
# BOOSTING: 0.7769929, 0.7824351, 0.778744, 0.7726337 : 0.7777014
# GLMNET: 0.7152989, 0.6911765, 0.7054583, 0.609589 : 0.6803807
# MAXENT: 0.8288016, 0.8018109, 0.8171717, 0.8455523 : 0.8233341
# NNET: Library Bug: N/A
# RF: 0.9169169, 0.8904899, 0.9039039, 0.9281998 : 0.9098776
# SVM: 0.8921079, 0.8843212, 0.8843750, 0.8590131 : 0.8799543
# TREE: Library Bug: N/A


# RF Selected. Now build model for first level categories (with final.sparse.const set to 0.998).

first.level.model = train_model(container, "RF")

# Save the test result of first level classifier
first.level.result = classify_model(container, first.level.model)[,1]
first.level.correct.result = as.factor(combine.first.test.targets)
first.level.correct.p = sum(first.level.result == first.level.correct.result) / length(first.level.correct.result)
# 0.9765 - High enough. Let's continue.





# Step 2: Create Classifiers within First Level Categories (and use them to classify all data)

for (fcid in 1:length(first.level.cats)) {
  cat = first.level.cats[fcid]
  
  # comp.combine.tdm = as.DocumentTermMatrix(c(comp.train.tdm, combine.test.tdm))
  assign(paste0(cat, ".combine.tdm"), as.DocumentTermMatrix(c(get(paste0(cat, ".train.tdm")), combine.test.tdm)))
  # comp.combine.targets = c(comp.train.targets, combine.test.targets)
  assign(paste0(cat, ".combine.targets"), c(get(paste0(cat, ".train.targets")), combine.second.test.targets))
  # temp.container = create_container(comp.combine.tdm, comp.combine.targets, trainSize=1:length(comp.train.targets), testSize=(length(comp.train.targets)+1):length(comp.combine.targets), virgin=FALSE)
  temp.container = create_container(get(paste0(cat, ".combine.tdm")), get(paste0(cat, ".combine.targets")), trainSize=1:length(get(paste0(cat, ".train.targets"))), testSize=(length(get(paste0(cat, ".train.targets")))+1):length(get(paste0(cat, ".combine.targets"))), virgin=FALSE)
  # comp.model = train_model(temp.container, "RF")
  assign(paste0(cat, ".model"), train_model(temp.container, "RF"))
  # comp.second.result = classify_model(temp.container, comp.model)
  assign(paste0(cat, ".second.result"), classify_model(temp.container, get(paste0(cat, ".model")))[,1])
}

correct.count = 0

for (i in 1:length(first.level.correct.result)) {
  if (get(paste0(first.level.cats[first.level.result[i]], ".second.result"))[i] == combine.second.test.targets[i]) {
    correct.count = correct.count + 1
  }
}

multistep.correct.p = correct.count / length(first.level.correct.result)
# Final correct rate: 0.976.

