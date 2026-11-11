

```sh
python -m spacy download en_core_web_sm
```

# Punkt

Punkt is an unsupervised algorithm for sentence boundary detection, from Kiss and Strunk (2006). In NLTK, `punkt` is also the name of the data package containing pre-trained Punkt models, one per language (English, French, German, and so on).

**The problem it solves**
Splitting text into sentences looks trivial (split on `.`), but periods are ambiguous. They end sentences, but they also mark abbreviations (`Dr.`, `etc.`, `U.S.`), appear in numbers (`3.14`), and sit inside initials (`J. R. R. Tolkien`). Punkt decides which periods are real sentence boundaries.

```python
import nltk
nltk.download('punkt_tab')   # newer NLTK versions
# nltk.download('punkt')     # older versions

from nltk.tokenize import sent_tokenize
sent_tokenize("Dr. Smith arrived. He was late.")
# ['Dr. Smith arrived.', 'He was late.']
```

**nltk**
One thing worth knowing given your francophone terminology work: `word_tokenize` accepts `language='french'`, but that only changes the sentence-splitting model. The word-level rules are still English-oriented Treebank rules, so contractions like `l'étudiant` or `qu'il` won't be handled the French way. For French, a regex tokenizer you control, or spaCy's French model, is usually a better fit.
