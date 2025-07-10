import ollama
import pandas as pd
import numpy as np
import random
from transformers import BertTokenizer, AutoModel
from sklearn.utils import shuffle
import torch
import torch.nn.functional as F

model_name = 'indobenchmark/indobert-lite-base-p2'
model = AutoModel.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)
model.eval()

def ganti_kata_dengan_sinonim(kalimat_asli):
    prompt = f"""
    Ganti beberapa kata secara acak dari kalimat berikut dengan sinonim yang sesuai konteks (tanpa mengubah arti kalimat). 
    - Jangan ubah struktur kalimat atau makna.
    - Jangan ganti nama orang, tempat, atau istilah khusus.

    Kalimat: {kalimat_asli}
    Kalimat hasil:"""

    response = ollama.chat(
        model='gemma3',  # Ganti dengan model yang kamu gunakan di Ollama
        messages=[{'role': 'user', 'content': prompt}]
    )

    return response['message']['content'].strip()

data_train = ["data/clean/spesific/"]

for data in data_train:
    df = pd.read_csv(data+"train_indo.csv")
    df['before'] = df['answer']
    print(df.info())
    print(len(df['score'].unique()))

    # create bin
    bins = np.arange(0, 110, 10)
    labels = [f"{i}-{i+9}" for i in bins[:-1]]
    df['score_bin'] = pd.cut(df['score'], bins=bins, labels=labels, include_lowest=True)
    bin_value_count = df['score_bin'].value_counts()

    aug_data = []
    aug_answer = set()
    ban_question = ["analisis_essay-24", "analisis_essay-22", "analisis_essay-13", "analisis_essay-37", "analisis_essay-38"]
    random.seed(42)
    np.random.seed(42)
    max_count = max(bin_value_count)
    lower_bound = int(max_count * 0.9)
    
    for bin_label in labels:
        max_bin_value = random.randint(lower_bound, max_count)
        bin_data = df[df['score_bin'] == bin_label]
        needed_bin = max_bin_value - len(bin_data)

        if(needed_bin <= 0):
            continue

        while(needed_bin != 0):
            row = bin_data.sample(n=1).iloc[0]

            # skip synonym replacement for some question
            if row['dataset_num'] in ban_question:
                continue

            try:
                augmented_answer = ganti_kata_dengan_sinonim(row['answer'])
            except Exception as e:
                print(f"Gagal augmentasi: {e}")
                continue

            encoded_input = tokenizer([row['answer'], augmented_answer], max_length=512, padding='max_length', truncation=True, return_tensors='pt')
            with torch.no_grad():
                outputs = model(**encoded_input)
                cls_embeddings = outputs.last_hidden_state[:, 0, :]

            similarity = F.cosine_similarity(cls_embeddings[0], cls_embeddings[1], dim=0)
            if similarity < 0.95 or augmented_answer in aug_answer:
                continue

            aug_answer.add(augmented_answer)

            aug_data.append({
                'question': row['question'],
                'reference_answer': row['reference_answer'],
                'before': row['answer'],
                'answer': augmented_answer,
                'score': row['score'],
                'normalized_score': row['normalized_score'],
                'dataset': row['dataset'],
                'dataset_num': row['dataset_num'],
                'score_bin': row['score_bin']
            })

            needed_bin -= 1

    # Gabungkan data asli + augmentasi
    df_augmented = pd.DataFrame(aug_data)
    df_combined = shuffle(pd.concat([df, df_augmented], ignore_index=True))
    print(df_combined['score_bin'].value_counts())

    # Simpan jika perlu
    df_augmented.to_csv(f"{data}train_indo_balanced.csv", index=False)
    df_combined.to_csv(f"{data}train_indo_balanced_combined.csv", index=False)
    print("Augmentasi selesai. Dataset seimbang telah disimpan.")