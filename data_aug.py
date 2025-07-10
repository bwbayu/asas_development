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
        model='gemma3',
        messages=[{'role': 'user', 'content': prompt}]
    )

    return response['message']['content'].strip()

data_train = ["data/clean/spesific/"]

for data in data_train:
    df = pd.read_csv(data+"train_indo.csv")
    df['before'] = df['answer']
    print(df.info())
    print(len(df['score'].unique()))

    # buat bin dengan jarak 10
    bins = np.arange(0, 110, 10)
    # buat label bin
    labels = [f"{i}-{i+9}" for i in bins[:-1]]
    # mengelompokkan score ke bin
    df['score_bin'] = pd.cut(df['score'], bins=bins, labels=labels, include_lowest=True)
    # total data per bin
    bin_value_count = df['score_bin'].value_counts()

    aug_data = []
    # menghindari data duplikat
    aug_answer = set()
    # pertanyaan yang di skip
    ban_question = ["analisis_essay-24", "analisis_essay-22", "analisis_essay-13", "analisis_essay-37", "analisis_essay-38"]
    random.seed(42)
    np.random.seed(42)
    # mencari nilai max dari total data semua bin sebagai upper bound
    max_count = max(bin_value_count)
    # setting 90% dari nilai max sebagai lower bound
    lower_bound = int(max_count * 0.9)
    
    # loop per bin
    for bin_label in labels:
        # ambil nilai random antara lower dan upper bound
        # untuk menentukan bin tersebut total datanya berapa (maksimal jumlah data augmentasi)
        max_bin_value = random.randint(lower_bound, max_count)
        # ambil data bin terkait
        bin_data = df[df['score_bin'] == bin_label]
        # mencari jumlah data untuk augmentasi
        needed_bin = max_bin_value - len(bin_data)

        # skip
        if(needed_bin <= 0):
            continue

        # iterasi sampai jumlahnya terpenuhi
        while(needed_bin != 0):
            # ambil data random
            row = bin_data.sample(n=1).iloc[0]

            # filter pertanyaan
            if row['dataset_num'] in ban_question:
                continue
            
            # proses penggantian sinonim
            try:
                augmented_answer = ganti_kata_dengan_sinonim(row['answer'])
            except Exception as e:
                print(f"Gagal augmentasi: {e}")
                continue
            
            # ubah data sebelum dan sesudah augmentasi menjadi vektor
            encoded_input = tokenizer([row['answer'], augmented_answer], max_length=512, padding='max_length', truncation=True, return_tensors='pt')
            with torch.no_grad():
                outputs = model(**encoded_input)
                cls_embeddings = outputs.last_hidden_state[:, 0, :]

            # menghitung similarity antara data asli dengan augmentasi
            similarity = F.cosine_similarity(cls_embeddings[0], cls_embeddings[1], dim=0)

            # kalau similaritynya kurang dari 95% atau jawabannya duplikat maka di skip
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

    # concat data bin dan data asli
    df_augmented = pd.DataFrame(aug_data)
    df_combined = shuffle(pd.concat([df, df_augmented], ignore_index=True))
    print(df_combined['score_bin'].value_counts())

    # save data
    df_augmented.to_csv(f"{data}train_indo_balanced.csv", index=False)
    df_combined.to_csv(f"{data}train_indo_balanced_combined.csv", index=False)
    print("Augmentasi selesai. Dataset seimbang telah disimpan.")