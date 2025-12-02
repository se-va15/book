import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.decomposition import PCA, TruncatedSVD
from catboost import CatBoostRegressor
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.sparse import coo_matrix
from tqdm.auto import tqdm
import os
import gc
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

DATA_PATH = Path(__file__).parent


def smart_read(filename):
    path = DATA_PATH / filename
    try:
        df = pd.read_csv(path, sep=';', on_bad_lines='skip')
        if df.shape[1] < 2:
            df = pd.read_csv(path, sep=',', on_bad_lines='skip')
    except:
        df = pd.read_csv(path, sep=',', on_bad_lines='skip')
    df.columns = df.columns.str.strip()
    return df


print("Loading data...")
train_raw = smart_read('train.csv')
test_df = smart_read('test.csv')
users_df = smart_read('users.csv')
books_df = smart_read('books.csv')
book_genres_df = smart_read('book_genres.csv')
desc_df = smart_read('book_descriptions.csv')

print(f"Data loaded: train={len(train_raw)}, test={len(test_df)}")

books_df = books_df.drop_duplicates(subset=['book_id'])

try:
    MODEL_NAME = 'cointegrated/rubert-tiny2'
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)


    def get_embeddings(texts, batch_size=256):
        all_embeddings = []
        texts = [str(t) if pd.notna(t) and len(str(t)) > 0 else "" for t in texts]
        for i in tqdm(range(0, len(texts), batch_size), desc="BERT Inference"):
            batch = texts[i:i + batch_size]
            encoded = tokenizer(batch, padding=True, truncation=True, max_length=64, return_tensors='pt')
            with torch.no_grad():
                encoded = {k: v.to(device) for k, v in encoded.items()}
                output = model(**encoded)
                all_embeddings.append(output.last_hidden_state[:, 0, :].cpu().numpy())
        return np.vstack(all_embeddings)


    desc_uniq = desc_df.drop_duplicates(subset=['book_id'])
    embeddings = get_embeddings(desc_uniq['description'].tolist())

    pca = PCA(n_components=24, random_state=42)
    embeddings_pca = pca.fit_transform(embeddings)

    emb_cols = [f'bert_{i}' for i in range(24)]
    emb_df = pd.DataFrame(embeddings_pca, columns=emb_cols)
    emb_df['book_id'] = desc_uniq['book_id'].values
    emb_df['desc_len'] = desc_uniq['description'].fillna('').apply(len).values

    del model, tokenizer, embeddings
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    books_df = books_df.merge(emb_df, on='book_id', how='left')
    for col in emb_cols + ['desc_len']:
        books_df[col] = books_df[col].fillna(0)
    books_df = books_df.drop_duplicates(subset=['book_id'])

    print("BERT embeddings created successfully")

except Exception as e:
    print(f"BERT embeddings failed, using fallback: {e}")
    emb_cols = []
    books_df['desc_len'] = desc_df.groupby('book_id')['description'].fillna('').apply(len).reset_index(drop=True)
    books_df['desc_len'] = books_df['desc_len'].fillna(0)

all_users = np.unique(np.concatenate([train_raw['user_id'], test_df['user_id']]))
all_books = np.unique(np.concatenate([train_raw['book_id'], books_df['book_id']]))
user_map = {uid: i for i, uid in enumerate(all_users)}
book_map = {bid: i for i, bid in enumerate(all_books)}


def fit_predict_svd(train_data, val_data):
    rows = train_data['user_id'].map(user_map).values
    cols = train_data['book_id'].map(book_map).values
    data = train_data['rating'].values
    matrix = coo_matrix((data, (rows, cols)), shape=(len(all_users), len(all_books)))

    svd = TruncatedSVD(n_components=20, random_state=42)
    user_vecs = svd.fit_transform(matrix)
    book_vecs = svd.components_.T

    val_rows = val_data['user_id'].map(user_map).fillna(-1).astype(int).values
    val_cols = val_data['book_id'].map(book_map).fillna(-1).astype(int).values
    valid_mask = (val_rows != -1) & (val_cols != -1)

    result = np.full(len(val_data), train_data['rating'].mean())
    if np.any(valid_mask):
        result[valid_mask] = np.sum(user_vecs[val_rows[valid_mask]] * book_vecs[val_cols[valid_mask]], axis=1)
    return result


train_df = train_raw[train_raw['has_read'] == 1].reset_index(drop=True)
train_df['svd_feature'] = 0.0

print("Creating SVD features...")
kf_svd = KFold(n_splits=5, shuffle=True, random_state=42)
for t_idx, v_idx in tqdm(kf_svd.split(train_df), total=5):
    train_df.loc[v_idx, 'svd_feature'] = fit_predict_svd(train_df.iloc[t_idx], train_df.iloc[v_idx])

test_svd = fit_predict_svd(train_df, test_df)
test_df['svd_feature'] = test_svd

train_df['timestamp'] = pd.to_datetime(train_df['timestamp'], errors='coerce')
train_df = train_df.sort_values('timestamp')

first_genre = book_genres_df.groupby('book_id')['genre_id'].first().reset_index()
books_df = books_df.merge(first_genre, on='book_id', how='left')
books_df['genre_id'] = books_df['genre_id'].fillna(-1).astype(int)
books_df = books_df.drop_duplicates(subset=['book_id'])

train_enriched = train_df.merge(books_df[['book_id', 'author_id', 'genre_id']], on='book_id', how='left')


def add_history(df, grp, tgt):
    grouped = df.groupby(grp)[tgt]
    return (grouped.cumsum() - df[tgt]) / grouped.cumcount()


print("Creating historical features...")
train_enriched['user_hist'] = add_history(train_enriched, ['user_id'], 'rating')
train_enriched['user_genre_hist'] = add_history(train_enriched, ['user_id', 'genre_id'], 'rating')
train_enriched['user_author_hist'] = add_history(train_enriched, ['user_id', 'author_id'], 'rating')
train_enriched['book_hist'] = add_history(train_enriched, ['book_id'], 'rating')

g_mean = train_df['rating'].mean()
cols_hist = ['user_hist', 'user_genre_hist', 'user_author_hist', 'book_hist']
for c in cols_hist:
    train_enriched[c] = train_enriched[c].fillna(g_mean)

u_final = train_enriched.groupby('user_id')['rating'].mean().rename('user_hist')
ug_final = train_enriched.groupby(['user_id', 'genre_id'])['rating'].mean().rename('user_genre_hist')
ua_final = train_enriched.groupby(['user_id', 'author_id'])['rating'].mean().rename('user_author_hist')
b_final = train_enriched.groupby('book_id')['rating'].mean().rename('book_hist')

book_pop = train_raw.groupby('book_id').size().reset_index(name='book_pop')
user_act = train_raw.groupby('user_id').size().reset_index(name='user_act')


def enrich(base):
    d = base.merge(users_df, on='user_id', how='left')
    d = d.merge(books_df, on='book_id', how='left')
    d = d.merge(book_pop, on='book_id', how='left').fillna(0)
    d = d.merge(user_act, on='user_id', how='left').fillna(0)
    return d


print("Preparing final datasets...")
train_full = enrich(train_df)
for c in cols_hist:
    train_full[c] = train_enriched[c].values

test_full = enrich(test_df)
test_full = test_full.merge(u_final, on='user_id', how='left')
test_full = test_full.merge(ug_final, on=['user_id', 'genre_id'], how='left')
test_full = test_full.merge(ua_final, on=['user_id', 'author_id'], how='left')
test_full = test_full.merge(b_final, on='book_id', how='left')
for c in cols_hist:
    test_full[c] = test_full[c].fillna(g_mean)

cat_features = ['user_id', 'book_id', 'author_id', 'publisher', 'genre_id', 'gender', 'language']

for c in cat_features:
    train_full[c] = train_full[c].fillna(-1).astype(int)
    test_full[c] = test_full[c].fillna(-1).astype(int)

features = [
               'user_id', 'book_id',
               'author_id', 'publisher', 'genre_id',
               'age', 'gender', 'publication_year', 'language',
               'book_pop', 'user_act', 'desc_len',
               'user_hist', 'user_genre_hist', 'user_author_hist', 'book_hist',
               'svd_feature'
           ] + emb_cols

X = train_full[features]
y = train_full['rating']
X_test = test_full[features]

print(f"Final features: {len(features)}")
print(f"Train shape: {X.shape}, Test shape: {X_test.shape}")

print("\nStarting cross-validation...")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
test_preds = np.zeros(len(X_test))
scores = []

cat_params = {
    'iterations': 2000,
    'learning_rate': 0.05,
    'grow_policy': 'Lossguide',
    'max_leaves': 64,
    'l2_leaf_reg': 10,
    'loss_function': 'RMSE',
    'random_seed': 42,
    'verbose': 0,
    'early_stopping_rounds': 100,
    'allow_writing_files': False
}

lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'n_estimators': 2000,
    'learning_rate': 0.05,
    'num_leaves': 64,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'lambda_l2': 10,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1
}

for fold, (t_idx, v_idx) in enumerate(kf.split(X, y)):
    print(f"\nFold {fold + 1}")
    X_tr, y_tr = X.iloc[t_idx], y.iloc[t_idx]
    X_val, y_val = X.iloc[v_idx], y.iloc[v_idx]

    print("  Training CatBoost...")
    model_cat = CatBoostRegressor(**cat_params)
    model_cat.fit(X_tr, y_tr, cat_features=cat_features, eval_set=(X_val, y_val), use_best_model=True)
    p_cat = model_cat.predict(X_val)
    t_cat = model_cat.predict(X_test)

    print("  Training LightGBM...")
    model_lgb = lgb.LGBMRegressor(**lgb_params)
    callbacks = [lgb.early_stopping(stopping_rounds=100, verbose=False)]
    model_lgb.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric='rmse',
        categorical_feature=cat_features,
        callbacks=callbacks
    )
    p_lgb = model_lgb.predict(X_val)
    t_lgb = model_lgb.predict(X_test)

    p_blend = 0.5 * p_cat + 0.5 * p_lgb
    p_blend = np.clip(p_blend, 0, 10)

    score = 1 - (np.sqrt(mean_squared_error(y_val, p_blend)) / 10 + mean_absolute_error(y_val, p_blend) / 10) / 2
    scores.append(score)

    print(f"  Score: {score:.5f}")
    print(
        f"  CatBoost: {1 - (np.sqrt(mean_squared_error(y_val, p_cat)) / 10 + mean_absolute_error(y_val, p_cat) / 10) / 2:.4f}")
    print(
        f"  LightGBM: {1 - (np.sqrt(mean_squared_error(y_val, p_lgb)) / 10 + mean_absolute_error(y_val, p_lgb) / 10) / 2:.4f}")

    test_preds += (np.clip(t_cat, 0, 10) + np.clip(t_lgb, 0, 10)) / 2

print(f"\nAverage Ensemble Score: {np.mean(scores):.5f}")

submission = pd.DataFrame({
    'user_id': test_full['user_id'],
    'book_id': test_full['book_id'],
    'rating_predict': test_preds / 5
})

submission_path = DATA_PATH / 'submission.csv'
submission.to_csv(submission_path, index=False)
print(f"\n✅ Submission saved to: {submission_path}")
print(f"📊 Predictions range: {submission['rating_predict'].min():.2f} - {submission['rating_predict'].max():.2f}")