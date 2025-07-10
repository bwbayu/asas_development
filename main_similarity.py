# ============================================================================= DATASET
import torch
from torch.utils.data import Dataset
import re

class AutomaticScoringDataset(Dataset):
    def __init__(self, dataframe, tokenizer):
        self.dataframe = dataframe
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataframe)

    def preprocess_text(self, text):
        text = text.lower()  # Ubah ke lowercase
        text = re.sub(r"[^a-zA-Z0-9\s]", ' ', text)  # Hapus karakter khusus
        text = ' '.join(text.split())  # Hapus spasi berlebih
        return text

    def __getitem__(self, index):
        student_answer = self.preprocess_text(str(self.dataframe.iloc[index]['answer']))
        if 'normalized_score' in self.dataframe.columns:
            score = self.dataframe.iloc[index]['normalized_score']
        else:
            score = self.dataframe.iloc[index]['score'] / 100.0
        reference_answer = self.preprocess_text(str(self.dataframe.iloc[index]['reference_answer']))
        
        # encode
        encoding_reference = self.tokenizer.encode_plus(
            reference_answer,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        encoding_student = self.tokenizer.encode_plus(
            student_answer,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        encoding_reference = {key: tensor.squeeze(0) for key, tensor in encoding_reference.items()}
        encoding_student = {key: tensor.squeeze(0) for key, tensor in encoding_student.items()}
        labels = torch.tensor(score, dtype=torch.float)

        return {
            'reference_answer': encoding_reference,
            'student_answer': encoding_student,
            'labels': labels
        }
    
# ============================================================================= MODEL
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from sentence_transformers import SentenceTransformer

class SiameseModel(nn.Module):
    def __init__(self, model_name='sentence-transformers/distiluse-base-multilingual-cased-v2', hidden_dropout=0.1, attention_dropout=0.1, pooling_type='mean'):
        super().__init__()
        # init pretrained
        self.pooling_type = pooling_type
        self.model = SentenceTransformer(model_name)
        self.config = AutoConfig.from_pretrained(model_name)
        self.config.dropout = hidden_dropout
        self.config.attention_dropout = attention_dropout
        self.encoder = AutoModel.from_pretrained(model_name, config=self.config)
        
        set_seed(SEED)
        # add other component
        self.dense = nn.Linear(768, 512)
        # copy weight from pretrained model
        with torch.no_grad():
          self.dense.weight.copy_(self.model[2].linear.weight)
          self.dense.bias.copy_(self.model[2].linear.bias)
        self.activation = nn.Tanh()

        if pooling_type == "attention":
            self.query_vector = nn.Linear(self.config.hidden_size, 1)

    def mean_pooling(self, token_embeddings, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1).clamp(min=1e-9)
        return sum_embeddings / sum_mask  # Mean pooling
    
    def max_pooling(self, token_embeddings, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        token_embeddings[input_mask_expanded == 0] = -1e9
        return torch.max(token_embeddings, 1)[0]

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state

        if(self.pooling_type == 'mean'):
            cls_embedding = self.mean_pooling(hidden_states, attention_mask)
        elif(self.pooling_type == 'max'):
            cls_embedding = self.max_pooling(hidden_states, attention_mask)
        elif self.pooling_type == "attention":
            attn_scores = self.query_vector(hidden_states).squeeze(-1)
            if attention_mask is not None:
                attn_scores = attn_scores.masked_fill(attention_mask == 0, -1e9)
            attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(-1)
            cls_embedding = torch.sum(hidden_states * attn_weights, dim=1)
        else:
            cls_embedding = hidden_states[:, 0, :]

        x = self.dense(cls_embedding)                                   # [B, 512]
        x = self.activation(x)
        return x

# ============================================================================= PIPELINE
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, BertTokenizer
from torch.utils.data import DataLoader
import torch
from torch.optim import AdamW
from src.utils.EarlyStopping import EarlyStopping
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr
import time
import logging
import torch.nn.functional as F
from sklearn.linear_model import LinearRegression
import joblib

SEED = 42
torch.manual_seed(SEED)

# logging setup
logging.basicConfig(
    filename="training.log",
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

# init device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BERTPipeline:
    def __init__(self, config, results, results_epoch):
        set_seed(SEED)
        self.train_df = config['train_df']
        self.valid_df = config['valid_df']
        self.test_df = config['test_df']
        # tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
        self.model = SiameseModel(config['model_name'],
            attention_dropout=config['attention_dropout'],
            hidden_dropout=config['hidden_dropout'],
            pooling_type=config['pooling_type']
            ).to(device)

        print(self.model)
        self.reg = LinearRegression()
        # optimizer and scheduler
        self.optimizer = AdamW(self.model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        # early stopping
        self.early_stopping = EarlyStopping(patience=5, verbose=True, path='experiments/models/checkpoint.pt')
        # loss function
        self.criterion = torch.nn.MSELoss()
        # other variable
        self.config = config
        self.results = results
        self.results_epoch = results_epoch
    
    def create_dataset(self, train_dataset, valid_dataset, test_dataset):
        print("create dataset run...")
        train_data = AutomaticScoringDataset(train_dataset, self.tokenizer)
        valid_data = AutomaticScoringDataset(valid_dataset, self.tokenizer)
        test_data = AutomaticScoringDataset(test_dataset, self.tokenizer)

        return train_data, valid_data, test_data
    
    def create_dataloader(self, train_data, valid_data, test_data):
        print("create dataloader run...")
        train_dataloader = DataLoader(train_data, batch_size=self.config['batch_size'], shuffle=True, generator=torch.Generator().manual_seed(SEED), num_workers=0)
        valid_dataloader = DataLoader(valid_data, batch_size=self.config['batch_size'], shuffle=False, generator=torch.Generator().manual_seed(SEED), num_workers=0)
        test_dataloader = DataLoader(test_data, batch_size=self.config['batch_size'], shuffle=False, generator=torch.Generator().manual_seed(SEED), num_workers=0)

        return train_dataloader, valid_dataloader, test_dataloader

    @staticmethod
    def save_model(model, save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")
        logging.info(f"Model saved to {save_path}")

    def compute_cosine_similarity(self, dataloader):
        all_outputs = []
        all_scores = []
        with torch.no_grad():
            for batchs in dataloader:
                batchs = {
                    k: {kk: vv.to(device) for kk, vv in v.items()} if isinstance(v, dict) else v.to(device)
                    for k, v in batchs.items()
                }
                
                # get embedding for each student and reference answer
                reference_emb = self.model(batchs['reference_answer']['input_ids'], batchs['reference_answer']['attention_mask'])
                student_emb = self.model(batchs['student_answer']['input_ids'], batchs['student_answer']['attention_mask'])
                scores = batchs['labels'].float().view(-1)

                # Normalize embeddings
                ref_embedding = F.normalize(reference_emb, p=2, dim=1)
                student_embedding = F.normalize(student_emb, p=2, dim=1)

                # get cosine similarity
                similarity = F.cosine_similarity(ref_embedding, student_embedding, dim=1)
                similarity = torch.clamp(similarity, -1.0, 1.0)

                all_outputs.append(similarity)
                all_scores.append(scores)

        # Concatenate all batches
        X = torch.cat(all_outputs, dim=0).cpu().numpy()
        y = torch.cat(all_scores, dim=0).cpu().numpy()

        return X, y


    def evaluate(self, dataloader, mode="validation", model_path=None):
        model_to_evaluate = self.model
        if mode == 'testing':
            if model_path and os.path.exists(model_path):
                model_to_evaluate = SiameseModel(self.config['model_name'],
                    attention_dropout=self.config['attention_dropout'],
                    hidden_dropout=self.config['hidden_dropout'],
                    pooling_type=self.config['pooling_type']
                    ).to(device)

                # model_to_evaluate = SiameseModel(self.config['model_name'],
                #                      attention_dropout=self.config['attention_dropout'],
                #                      hidden_dropout=self.config['hidden_dropout']).to(device)
                try:
                    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
                    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                        model_to_evaluate.load_state_dict(checkpoint['model_state_dict'])
                    else:
                        model_to_evaluate.load_state_dict(checkpoint)
                except Exception as e:
                    print(f"Error loading model from {model_path}: {e}")
                    logging.error(f"Error loading model from {model_path}: {e}")
                    print("Warning: Failed to load specified model path. Using current instance model state for testing.")
                    model_to_evaluate = self.model

        model_to_evaluate.eval()
        total_mse_loss = 0
        all_predictions = []
        all_targets = []
        with torch.no_grad():
            for batchs in dataloader:
                try:
                    # move to device
                    batchs = {
                        k: {kk: vv.to(device) for kk, vv in v.items()} if isinstance(v, dict) else v.to(device)
                        for k, v in batchs.items()
                    }
                    
                    # get embedding for each student and reference answer
                    reference_emb = model_to_evaluate(batchs['reference_answer']['input_ids'], batchs['reference_answer']['attention_mask'])
                    student_emb = model_to_evaluate(batchs['student_answer']['input_ids'], batchs['student_answer']['attention_mask'])
                    scores = batchs['labels'].float().view(-1)

                    # Normalize embeddings
                    ref_embedding = F.normalize(reference_emb, p=2, dim=1)
                    student_embedding = F.normalize(student_emb, p=2, dim=1)

                    # get cosine similarity
                    similarity = F.cosine_similarity(ref_embedding, student_embedding, dim=1)
                    similarity = torch.clamp(similarity, -1.0, 1.0)

                    # calculate loss
                    loss = self.criterion(similarity, scores)
                    if torch.isnan(loss):
                        print(f"Predictions: {similarity}")
                        print(f"Targets: {batchs['labels']}")
                        continue

                    total_mse_loss += loss.item()

                    all_predictions.extend(similarity.detach().cpu().numpy())
                    all_targets.extend(batchs['labels'].detach().cpu().numpy())
                except Exception as e:
                    logging.error(f"Error during {mode}: {str(e)}")
                    torch.cuda.empty_cache()

        avg_mse_loss = total_mse_loss / max(len(dataloader), 1)
        mae = mean_absolute_error(all_targets, all_predictions)
        rmse = np.sqrt(mean_squared_error(all_targets, all_predictions))
        pearson, _ = pearsonr(all_targets, all_predictions)

        return avg_mse_loss, mae, rmse, pearson
    
    def training(self):
        # create dataset
        train_data, valid_data, test_data = self.create_dataset(self.train_df, self.valid_df, self.test_df)
        train_dataloader, valid_dataloader, test_dataloader = self.create_dataloader(train_data, valid_data, test_data)

        # init start training time
        start_time = time.time()
        # experiment process
        epochs = self.config["epochs"]
        num_epochs = 0
        best_valid_metric = self.config["best_valid_rmse"] if self.config["best_valid_rmse"] is not None else float('inf')
        best_model_path = os.path.join("experiments", "models", "dropout", f"{self.config['split_type']}", f"sbert.pt")
        valid_metric_config = float('inf')
        model_path_config = os.path.join("experiments", "models", "dropout", f"{self.config['split_type']}", f"sbert_{self.config['config_id']}.pt")
        for epoch in range(epochs):
            num_epochs += 1
            print(f"====== Training Epoch {epoch + 1}/{epochs} ======")
            self.model.train()
            train_mse_loss = 0
            all_predictions = []
            all_targets = []
            for batchs in train_dataloader:
                try:
                    self.optimizer.zero_grad()
                    # move to device
                    batchs = {
                        k: {kk: vv.to(device) for kk, vv in v.items()} if isinstance(v, dict) else v.to(device)
                        for k, v in batchs.items()
                    }


                    # get embedding for each student and reference answer
                    reference_emb = self.model(batchs['reference_answer']['input_ids'], batchs['reference_answer']['attention_mask'])
                    student_emb = self.model(batchs['student_answer']['input_ids'], batchs['student_answer']['attention_mask'])
                    scores = batchs['labels'].float().view(-1)

                    # Normalize embeddings (normalize vector length == 1)
                    ref_embedding = F.normalize(reference_emb, p=2, dim=1)
                    student_embedding = F.normalize(student_emb, p=2, dim=1)

                    # get cosine similarity
                    similarity = F.cosine_similarity(ref_embedding, student_embedding, dim=1)
                    similarity = torch.clamp(similarity, -1.0, 1.0)

                    # calculate loss
                    loss = self.criterion(similarity, scores)
                    if torch.isnan(loss):
                        print(f"Predictions: {similarity}")
                        print(f"Targets: {scores}")
                        continue
                    
                    # backprop
                    loss.backward()
                    self.optimizer.step()

                    # save data for calculation
                    train_mse_loss += loss.item()
                    all_predictions.extend(similarity.detach().cpu().numpy())
                    all_targets.extend(scores.detach().cpu().numpy())
                except Exception as e:
                    logging.error(f"Error during training: {str(e)}")
                    torch.cuda.empty_cache()

            # calculate loss function and evaluation metrik
            avg_train_loss = train_mse_loss / max(len(train_dataloader), 1)
            train_mae = mean_absolute_error(all_targets, all_predictions)
            train_rmse = np.sqrt(mean_squared_error(all_targets, all_predictions))
            train_pearson, _ = pearsonr(all_targets, all_predictions)
            print(f"Epoch {epoch+1}/{epochs} - Avg training loss: {avg_train_loss:.4f}, MAE: {train_mae:.4}, RMSE: {train_rmse:.4}, Pearson Corr: {train_pearson:.4}")

            # EVALUATION PROCESS
            valid_loss, valid_mae, valid_rmse, valid_pearson = self.evaluate(valid_dataloader, mode="validation")
            print(f"Avg validation loss: {valid_loss:.4f}, MAE: {valid_mae:.4}, RMSE: {valid_rmse:.4}, Pearson Corr: {valid_pearson:.4}")

            # save experiment result per epoch
            self.results_epoch.append({
                "config_id": self.config["config_id"],
                "epoch": epoch + 1,
                "train_mse": avg_train_loss,
                "train_mae": train_mae,
                "train_rmse": train_rmse,
                "train_pearson": train_pearson,
                "valid_mse": valid_loss,
                "valid_mae": valid_mae,
                "valid_rmse": valid_rmse,
                "valid_pearson": valid_pearson
            })

            # check early stopping
            self.early_stopping(val_loss=valid_loss, model=self.model)
            if(self.early_stopping.early_stop):
                logging.info(f"Early stopping triggered")
                print("Early stopping triggered")
                break

            # save best model in every config
            if valid_rmse < best_valid_metric:
                best_valid_metric = valid_rmse
                self.save_model(self.model, save_path=best_model_path) 

            # save model for each config for testing
            if valid_rmse < valid_metric_config:
                valid_metric_config = valid_rmse
                self.save_model(self.model, save_path=model_path_config)
        
        # TESTING PROCESS WHEN FINE-TUNING SBERT IS NOT AVAILABLE ==== TEST DATA IN LINEAR REGRESSION
        # # TESTING PROCESS
        # test_loss, test_mae, test_rmse, test_pearson = self.evaluate(
        #     test_dataloader, mode="testing", model_path=model_path_config
        # )        
        # print(f"Avg testing loss: {test_loss:.4f}, MAE: {test_mae:.4}, RMSE: {test_rmse:.4}, Pearson Corr: {test_pearson:.4}")

        # LINEAR REGRESSION
        self.model.eval()
        X_train, y_train = self.compute_cosine_similarity(train_dataloader)
        X_valid, y_valid = self.compute_cosine_similarity(valid_dataloader)
        X_test, y_test = self.compute_cosine_similarity(test_dataloader)

        # TRAINING
        
        self.reg.fit(X_train.reshape(-1, 1), y_train)
        joblib.dump(self.reg, os.path.join("experiments", "models", "dropout", f"{self.config['split_type']}", f"reg_{self.config['config_id']}.pkl"))
        y_train_pred = self.reg.predict(X_train.reshape(-1, 1))
        reg_train_loss = mean_squared_error(y_train, y_train_pred)
        reg_train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        reg_train_pearson, _ = pearsonr(y_train, y_train_pred)

        # VALIDATION
        y_valid_pred = self.reg.predict(X_valid.reshape(-1, 1))
        reg_valid_loss = mean_squared_error(y_valid, y_valid_pred)
        reg_valid_rmse = np.sqrt(mean_squared_error(y_valid, y_valid_pred))
        reg_valid_pearson, _ = pearsonr(y_valid, y_valid_pred)

        # TESTING
        y_test_pred = self.reg.predict(X_test.reshape(-1, 1))
        reg_test_loss = mean_squared_error(y_test, y_test_pred)
        reg_test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        reg_test_pearson, _ = pearsonr(y_test, y_test_pred)

        # save experiment per configuration
        result = {
            "config_id": self.config.get("config_id"),
            "type_test": self.config.get("type_test"),
            "model_name": self.config.get("model_name"),
            # # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! CHANGE THIS
            # "dataset_type": "after balancing", 
            "pooling_type": self.config.get("pooling_type"), 
            "batch_size": self.config.get("batch_size"),
            "epochs": num_epochs,
            "learning_rate": self.config.get("learning_rate"),
            "attention_dropout": self.config['attention_dropout'],
            "hidden_dropout": self.config['hidden_dropout'],
            "weight_decay": self.config['weight_decay'],
            "training_time": time.time() - start_time,
            "peak_memory": torch.cuda.max_memory_allocated(device) / (1024 ** 2),  # Convert to MB
            "reg_train_loss": reg_train_loss,
            "reg_train_rmse": reg_train_rmse,
            "reg_train_pearson": reg_train_pearson,
            "reg_valid_loss": reg_valid_loss,
            "reg_valid_rmse": reg_valid_rmse,
            "reg_valid_pearson": reg_valid_pearson,
            "reg_test_loss": reg_test_loss,
            "reg_test_rmse": reg_test_rmse,
            "reg_test_pearson": reg_test_pearson
        }

        # Tambahkan hasil ke dalam list results
        self.results.append(result)

    @staticmethod
    def save_csv(data, filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        file_exists = os.path.exists(filename)
        pd.DataFrame(data).to_csv(
            filename, mode="a" if file_exists else "w", header=not file_exists, index=False
        )

# ============================================================================= MAIN
import pandas as pd
import logging
import torch
import os
import random
import numpy as np
import transformers

def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
SEED = 42

def main():
    SEED = 42
    transformers.logging.set_verbosity_error()
    set_seed(SEED)
    ROOT_DIR = os.getcwd()
    split_type = ['spesific', 'cross']
    for split in split_type:
        dataset_list = [
        (f"data/clean/{split}/train_indo_balanced.csv", f"data/clean/{split}/valid_indo.csv", f"data/clean/{split}/test_indo.csv", "indo", "sentence-transformers/distiluse-base-multilingual-cased-v2")
        ]
        df_exp = pd.read_csv("data/tabel_eksperimen_similarity.csv")

        experiments = []

        for _, row in df_exp.iterrows():
            experiments.append({
                "attention_dropout": row["attention_dropout"],
                "hidden_dropout": row["hidden_dropout"],
                "type_test": row["type_test"],
                "weight_decay": row["weight_decay"],
                "pooling_type": row["pooling_type"]
            })

        print("Total Experiment : ", len(experiments))
        for train_data, valid_data, test_data, dataset_type, model_name in dataset_list:
            train_df = pd.read_csv(train_data)
            print(train_df.info())

            valid_df = pd.read_csv(valid_data)
            print(valid_df.info())

            test_df = pd.read_csv(test_data)
            print(test_df.info())
            for config in experiments:
                set_seed(SEED)
                # Check if the first file exists
                df_result = None
                results = []
                path = f"experiments/results/dropout/{split}/sbert.csv"
                epoch_path = f"experiments/results/dropout/{split}/sbert_epoch.csv"
                if os.path.exists(path):
                    df_result = pd.read_csv(path)
                    print(df_result['config_id'].iloc[-1])
                else:
                    print(f"File 'sbert.csv' does not exist.")

                idx = (df_result['config_id'].iloc[-1] + 1) if df_result is not None and not df_result.empty else 0  # index untuk setiap kombinasi
                results_epoch = []
                df_result1 = None

                # Check if the second file exists
                if os.path.exists(epoch_path):
                    df_result1 = pd.read_csv(epoch_path)
                    print(min(df_result1['valid_rmse']))
                else:
                    print(f"File 'sbert_epoch.csv' does not exist.")

                run_config = {
                    "train_df": train_df,
                    "valid_df": valid_df,
                    "test_df": test_df,
                    "model_name": model_name,
                    "batch_size": 16,
                    "learning_rate": 2e-5,
                    "epochs": 100,
                    "config_id": idx,
                    "best_valid_rmse": min(df_result1['valid_rmse']) if df_result1 is not None and not df_result1.empty else float("inf"),
                    "dataset_type": dataset_type,
                    "attention_dropout": config["attention_dropout"],
                    "hidden_dropout": config["hidden_dropout"],
                    "weight_decay": config["weight_decay"],
                    "type_test": config["type_test"],
                    "pooling_type": config["pooling_type"],
                    "split_type": split
                }

                logging.info(
                    f"Running configuration: config_id={idx}, model_name={run_config['model_name']}, epochs={100}, type_test:{run_config['type_test']}"
                )

                print(
                    f"\nRunning configuration: config_id={idx}, model_name={run_config['model_name']}, epochs={100}, type_test:{run_config['type_test']}"
                )

                try:
                    pipeline = BERTPipeline(run_config, results, results_epoch)
                    pipeline.training()

                    # Save results
                    # Dapatkan root project
                    results_path = os.path.join(ROOT_DIR, path)
                    results_epoch_path = os.path.join(ROOT_DIR, epoch_path)
                    pipeline.save_csv(results, results_path)
                    pipeline.save_csv(results_epoch, results_epoch_path)
                except Exception as e:
                    logging.error(f"Error in config_id={idx}: {str(e)}")
                    print(f"Error in config_id={idx}: {str(e)}")
                    torch.cuda.empty_cache()
                finally:
                    # Clear GPU memory after every configuration
                    del pipeline.model
                    del pipeline.tokenizer
                    del pipeline.optimizer
                    del pipeline
                    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()