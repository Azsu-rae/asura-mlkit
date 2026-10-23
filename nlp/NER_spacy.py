
import spacy

nlp = spacy.load("en_core_web_sm")

sentence = "Why Apple is looking at buying Algerian startup for 1 billion dinars ?"
doc = nlp(sentence)

for ent in doc.ents:
    print(ent.text, ent.start_char, ent.end_char, ent.label_)
