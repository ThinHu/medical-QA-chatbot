import os
import sys
import json
import torch

# Add src to path so utils can be loaded
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from utils import load_tokenizer, MODEL_CLASSES, get_entity_label

class ViMQInferencer:
    def __init__(self, model_dir):
        args_path = os.path.join(model_dir, "training_args.bin")
        self.args = torch.load(args_path, weights_only=False)
        self.args.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.args.data_dir = os.path.join(os.path.dirname(__file__), "data")
        
        self.tokenizer = load_tokenizer(self.args)
        self.label2index, self.index2label = get_entity_label(self.args)
        
        self.config_class, self.model_class, _ = MODEL_CLASSES[self.args.model_type]
        self.config = self.config_class.from_pretrained(self.args.model_name_or_path)
        self.model = self.model_class(config=self.config, args=self.args)
        
        model_path = os.path.join(model_dir, "model.pt")
        self.model.load_state_dict(torch.load(model_path, map_location=self.args.device, weights_only=True))
        self.model.to(self.args.device)
        self.model.eval()
        
        char_vocab_path = os.path.join(self.args.data_dir, self.args.file_name_char2index)
        with open(char_vocab_path, 'r', encoding='utf-8') as f:
            self.char_vocab = json.load(f)

    def span_decode(self, logits):
        arg_index = []
        for i in range(len(logits)):
            for j in range(i, len(logits[i])):
                if logits[i][j] > 0:
                    arg_index.append([i, j, self.index2label.get(int(logits[i][j]), 'UNK')])
        return arg_index

    def predict(self, text: str):
        words = text.split()
        input_ids, attention_mask, firstSWindices = [], [], []
        
        input_ids.append(self.tokenizer.cls_token_id)
        attention_mask.append(1)
        
        for w in words:
            word_tokens = self.tokenizer.tokenize(w)
            if not word_tokens: continue
            
            firstSWindices.append(len(input_ids))
            for token in word_tokens:
                input_ids.append(self.tokenizer.convert_tokens_to_ids(token))
                attention_mask.append(1)
                
        input_ids.append(self.tokenizer.sep_token_id)
        attention_mask.append(1)
        
        max_seq_len = self.args.max_seq_len
        pad_len = max_seq_len - len(input_ids)
        input_ids += [self.tokenizer.pad_token_id] * pad_len
        attention_mask += [0] * pad_len
        
        firstSWindices += [0] * (max_seq_len - len(firstSWindices))
        
        char_ids = []
        max_char_len = self.args.max_char_len
        for w in words:
            char_seq = [self.char_vocab.get(c, self.char_vocab.get('UNK')) for c in w]
            char_seq = char_seq[:max_char_len]
            char_seq += [self.char_vocab.get('PAD', 0)] * (max_char_len - len(char_seq))
            char_ids.append(char_seq)
            
        char_pad_len = max_seq_len - len(words)
        char_ids += [[self.char_vocab.get('PAD', 0)] * max_char_len] * char_pad_len
        
        input_ids_t = torch.tensor([input_ids]).to(self.args.device)
        attention_mask_t = torch.tensor([attention_mask]).to(self.args.device)
        firstSWindices_t = torch.tensor([firstSWindices]).to(self.args.device)
        seq_len_t = torch.tensor([[len(words)]]).to(self.args.device)
        char_ids_t = torch.tensor([char_ids]).to(self.args.device)
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids_t,
                attention_mask=attention_mask_t,
                firstSWindices=firstSWindices_t,
                seq_len=seq_len_t,
                char_ids=char_ids_t
            )
            
            intent_logits = outputs[0]
            entity_logits = outputs[1]
            
            intent_idx = torch.argmax(intent_logits, dim=1).item()
            intent_label = getattr(self, "intent_index2label", {}).get(intent_idx, f"INTENT_{intent_idx}")
            
            entity_logits = entity_logits.cpu().numpy()[0]
            spans = self.span_decode(entity_logits)
            
            entities = []
            for span in spans:
                start, end, label = span
                if start < len(words) and end < len(words):
                    entity_text = " ".join(words[start:end+1])
                    entities.append(f"{entity_text} ({label})")
                    
        return {
            "intent": intent_label,
            "entities": entities
        }

# SageMaker Entry Point Functions

def model_fn(model_dir):
    print("Loading ViMQ model from", model_dir)
    return ViMQInferencer(model_dir)

def input_fn(request_body, request_content_type):
    if request_content_type == "application/json":
        data = json.loads(request_body)
        return data.get("inputs", "")
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    if not input_data:
        return {"intent": "unknown", "entities": []}
    return model.predict(input_data)

def output_fn(prediction, accept):
    if accept == "application/json":
        return json.dumps(prediction)
    else:
        raise ValueError(f"Unsupported accept type: {accept}")
