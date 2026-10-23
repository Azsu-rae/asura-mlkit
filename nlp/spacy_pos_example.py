

import spacy

nlp = spacy.load("en_core_web_sm")
text = "SpaCy is a popular natural language processing library."

doc = nlp(text)

print("Original text: ", text)

print("PoS tagging Result:")
for token in doc:
    print(f"{token.text}: { token.pos_ }")
