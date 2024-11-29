# First, let's install required packages
import torch
import transformers
from transformers import BertForTokenClassification, BertTokenizer
import json
import os

class BertNERWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids, attention_mask)
        return outputs.logits

def prepare_bert_ner():
    # We'll use a pre-trained BERT model fine-tuned for NER
    model_name = "dslim/bert-base-NER"
    
    # Download model and tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForTokenClassification.from_pretrained(model_name)
    
    # Wrap the model
    wrapped_model = BertNERWrapper(model)
    wrapped_model.eval()
    
    # Create directory for exported files
    os.makedirs("exported_model", exist_ok=True)
    
    # Save tokenizer vocabulary and config
    tokenizer.save_pretrained("exported_model")
    
    # Save the ID to label mapping
    label_map = model.config.id2label
    with open("exported_model/label_map.json", "w") as f:
        json.dump(label_map, f)
    
    # Create an example input for tracing
    example_text = "John works at Microsoft in Seattle"
    inputs = tokenizer(example_text, 
                      return_tensors="pt",
                      padding=True,
                      truncation=True,
                      max_length=128)
    
    # Trace the wrapped model
    traced_model = torch.jit.trace(
        wrapped_model,
        (inputs['input_ids'], inputs['attention_mask'])
    )
    
    # Save the traced model
    traced_model.save("exported_model/traced_model.pt")
    
    print("Model exported successfully!")

def test_inference():
    model_name = "dslim/bert-base-NER"
    model = BertForTokenClassification.from_pretrained(model_name)
    tokenizer = BertTokenizer.from_pretrained(model_name)
    input_text = "John works at Microsoft in Seattle"
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    print(tokenizer.tokenize(input_text))
    outputs = model(inputs['input_ids'], inputs['attention_mask'])
    predictions = torch.argmax(outputs.logits, dim=2)
    map_predictions = [model.config.id2label[p.item()] for p in predictions[0]]
    print(map_predictions)
    
if __name__ == "__main__":
    #prepare_bert_ner()
    test_inference()