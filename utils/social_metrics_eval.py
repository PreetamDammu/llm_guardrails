from convokit import PolitenessStrategies
import spacy
import math
from detoxify import Detoxify

spacy_nlp = spacy.load('en_core_web_sm', disable=['ner'])
ps = PolitenessStrategies(strategy_collection="politeness_local", verbose=1000)

#LR coefficients from https://www.cs.cornell.edu/~cristian/Politeness_Paraphrasing_files/fine-grained-politeness-paraphrasing.pdf
lr_coefs = {'Actually':-0.358,'Adverb.Just':-0.004,'Affirmation':0.171,'Apology':0.429,'By.The.Way':0.331,'Conj.Start':-.0245,'Filler':-0.245,'For.Me':0.128,
 'For.You':0.197,'Gratitude':0.989,'Greeting':0.491,'Hedges':0.131,'Indicative':0.221,'Please':0.230,'Please.Start':-0.209,'Reassurance':0.668,'Subjunctive':0.454,'Swearing':-1.30}

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def compute_politeness_lrcoef(utterance):
    """
    Compute the politeness score for an utterance using the logistic regression coefficients
    Returns weigthted sum of the politeness strategies scores and the logistic regression coefficients
    """
    res = ps.transform_utterance(utterance, spacy_nlp=spacy_nlp)
    ps_keys = list(lr_coefs.keys())
    score = []
    for i in range(len(ps_keys)):
        score.append(res.meta['politeness_strategies'][ps_keys[i]]*lr_coefs[ps_keys[i]])

    return sigmoid(sum(score))

def compute_social_metrics(utterance):
    """
    Compute the social metrics for an utterance using the detoxify package
    Returns the toxicity score
    """
    politeness_score = compute_politeness_lrcoef(utterance)
    res = Detoxify('original').predict(utterance)
    res['politeness_score'] = politeness_score
    return res