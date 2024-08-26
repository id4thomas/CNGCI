## Seting up spacy
https://github.com/huggingface/neuralcoref/issues/222

```
pip uninstall neuralcoref
git clone https://github.com/huggingface/neuralcoref.git
cd neuralcoref
pip install -r requirements.txt
pip install -e .
pip uninstall spacy
pip install spacy==2.3.0
python -m spacy download en
```