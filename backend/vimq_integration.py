import os
import sys
import torch
import json
import numpy as np

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from typing import Dict, Any, List

# Cấu hình path để import code từ thư mục src của ViMQ
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from utils import load_tokenizer, MODEL_CLASSES, get_entity_label

class ViMQInferencer:
    def __init__(self, model_dir: str = None, use_mock: bool = False):
        self.use_mock = use_mock
        if self.use_mock:
            print("[ViMQ Integration] Cảnh báo: Đang chạy ở chế độ MOCK (giả lập). Vui lòng cập nhật trọng số mô hình thực tế để chạy inference thật.")
            return

        print("[ViMQ Integration] Đang tải mô hình từ:", model_dir)
        
        args_path = os.path.join(model_dir, "training_args.bin")
        if not os.path.exists(args_path):
            raise FileNotFoundError(f"Không tìm thấy training_args.bin tại {args_path}")
            
        self.args = torch.load(args_path, weights_only=False)
        self.args.device = "cuda" if torch.cuda.is_available() else "cpu"
        # override data_dir to local path
        self.args.data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
        
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

    def predict(self, text: str) -> Dict[str, Any]:
        """
        Nhận vào câu truy vấn của người dùng, trả về Intent và tập Entities.
        """
        if self.use_mock:
            text_lower = text.lower()
            intent = "triage"
            entities = []
            
            symptoms = ["đau đầu", "sốt cao", "ho", "khó thở", "chóng mặt", "đau bụng"]
            for s in symptoms:
                if s in text_lower:
                    entities.append(f"{s} (SYMPTOM)")
                    intent = "severity"
            
            if "khám ở đâu" in text_lower or "khoa nào" in text_lower:
                intent = "method_diagnosis"
                
            return {
                "intent": intent,
                "entities": entities
            }
        
        # Real pipeline
        sentence = text.lower()
        words = sentence.split(' ')
        char_seq = []
        for word in words:
            word_seq = []
            for i in range(self.args.max_char_len):
                try:
                    char = word[i]
                except:
                    char = self.args.pad_char
                word_seq.append(char)
            char_seq.append(word_seq)

        input_ids = [self.tokenizer.cls_token_id]
        firstSWindices = [len(input_ids)]

        for word in words:
            word_token = self.tokenizer.encode(word)
            input_ids += word_token[1: (len(word_token) - 1)]
            firstSWindices.append(len(input_ids))

        firstSWindices = firstSWindices[: (len(firstSWindices) - 1)]
        input_ids.append(self.tokenizer.sep_token_id)

        attention_mask = [1] * len(input_ids)

        if len(input_ids) > self.args.max_seq_len:
            input_ids = input_ids[:self.args.max_seq_len]
            attention_mask = attention_mask[:self.args.max_seq_len]
            firstSWindices = firstSWindices[:self.args.max_seq_len]
        else:
            attention_mask = attention_mask + [0] * (self.args.max_seq_len - len(input_ids))
            input_ids = input_ids + [self.tokenizer.pad_token_id] * (self.args.max_seq_len - len(input_ids))
            firstSWindices = firstSWindices + [0]*(self.args.max_seq_len - len(firstSWindices))

        input_ids_t = torch.tensor([input_ids]).to(self.args.device)
        attention_mask_t = torch.tensor([attention_mask]).to(self.args.device)
        firstSWindices_t = torch.tensor([firstSWindices]).to(self.args.device)
        seq_len_t = torch.tensor([[len(words)]]).to(self.args.device)

        char_ids = []
        for word in char_seq:
            word_char_ids = []
            for char in word:
                if char not in self.char_vocab:
                    word_char_ids.append(self.char_vocab.get("UNK"))
                else:
                    word_char_ids.append(self.char_vocab.get(char))
            char_ids.append(word_char_ids)
        if len(char_ids) < self.args.max_seq_len:
            char_ids += [[self.char_vocab.get("PAD")]*self.args.max_char_len]*(self.args.max_seq_len - len(char_ids))
        else:
            char_ids = char_ids[:self.args.max_seq_len]
        char_ids_t = torch.tensor([char_ids]).to(self.args.device)

        with torch.no_grad():
            outputs, _ = self.model(input_ids_t, attention_mask_t, firstSWindices_t, seq_len_t, char_ids_t)
        
        outputs_ = outputs.cpu().numpy()
        outputs_ = np.argmax(outputs_, axis=-1)
        output_spans = self.span_decode(outputs_[0])
        
        entities = []
        for span in output_spans:
            start, end, label = span
            if start < len(words) and end < len(words) and start <= end:
                ent_text = " ".join(words[start:end+1])
                entities.append(f"{ent_text} ({label})")
        
        # Intent will be classified by LLM based on these entities and user query
        intent = "PENDING_LLM_CLASSIFICATION"

        return {"intent": intent, "entities": entities}

# Tạo một singleton instance
model_dir_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ViMQ_Model")
vimq_model = ViMQInferencer(model_dir=model_dir_path, use_mock=False)

def analyze_query(query: str) -> Dict[str, Any]:
    return vimq_model.predict(query)
