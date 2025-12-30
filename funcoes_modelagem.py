import numpy as np
import pandas as pd


def X_3d(df, janela):

    data = df.to_numpy()
    if data.ndim == 1: 
        data = data.reshape(-1, 1)

    X, y = [], [] 
    for i in range(len(df) - janela):
        X.append(data[i:i+janela, :]) # janela de features
        y.append(data[i+janela, 0]) # alvo = primeira coluna
    
    return np.array(X), np.array(y)

def X_2d(Xseq):
    n_samples = Xseq.shape[0]
    return Xseq.reshape(n_samples, -1)


def split_treino_valid_teste(X, y, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
    
    n = len(X)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    # o resto vai para teste
    n_test = n - n_train - n_val
    
    # índices
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
    X_test, y_test = X[n_train+n_val:], y[n_train+n_val:]
    
    return X_train, y_train, X_val, y_val, X_test, y_test



def gerar_tabela_melhores(tuner, num_top=5):
    # melhores trials
    trials = tuner.oracle.get_best_trials(num_trials=num_top)
    lista_resultados = []
    
    for i, trial in enumerate(trials):
        #valores dos hiperparâmetros
        hps = trial.hyperparameters.values
        n_layers = hps.get('num_layers')
        
        # limpa parâmetros de camadas que não foram realmente usadas
        for key in list(hps.keys()):
            if 'units_' in key or 'dropout_' in key:
                layer_idx = int(key.split('_')[1])
                if layer_idx >= n_layers:
                    hps[key] = np.nan
        
        # metricas de desempenho
        hps['score_val_loss'] = trial.score
        hps['ranking'] = i + 1
        lista_resultados.append(hps)
    
    df = pd.DataFrame(lista_resultados)
    # reorganiza as colunas
    cols = ['ranking', 'score_val_loss', 'num_layers'] + [c for c in df.columns if c not in ['ranking', 'score_val_loss', 'num_layers']]
    return df[cols]