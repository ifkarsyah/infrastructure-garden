---
title: BERT
---
BERT (Bidirectional Encoder Representations from Transformers) is a pre-trained model for natural language processing tasks.

```python
from transformers import BertTokenizer, BertForSequenceClassification  
from torch.nn.functional import softmax  
  
# Load pre-trained BERT model and tokenizer  
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')  
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')  
  
# Tokenize and classify text  
input_text = "Your input text here"  
tokens = tokenizer(input_text, return_tensors='pt')  
outputs = model(**tokens)  
probabilities = softmax(outputs.logits, dim=1)  
  
# Print class probabilities  
print(probabilities)
```