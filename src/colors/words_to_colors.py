import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

from word_embs import Model

def main(langs = ['en','fr', 'de']):

    # Step 1: Load the aligned embeddings for English and French
    for lang in langs:
        print(f"Loading {lang} RGB data...")
        rgb_df = pd.read_csv(f'data/{lang}.csv')

        print(f"Loading aligned {lang} embeddings...")
        model = Model(lang)
        print(f"{lang} embeddings loaded.")
        model = model.return_model()

        # Step 4: Prepare the data for training
        # Match embeddings with their corresponding RGB values
        color_to_rgb = dict(zip(rgb_df['color'], rgb_df[['r', 'g', 'b']].values))
        colors = rgb_df['color'].to_list()
        X = []
        y = []
        for i, color in enumerate(colors):
            if color in model:
                X.append(model[color])
                y.append(color_to_rgb[color] / 255)

        X = np.array(X)
        y = np.array(y)

        # Step 5: Define a single-layer MLP model
        mlp = MLPRegressor(hidden_layer_sizes=(30,), max_iter=1000, random_state=42)

        # Step 6: Train the MLP model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.02, random_state=42)
        mlp.fit(X_train, y_train)
        y_train_pred = mlp.predict(X_train)

        print(f"TRAIN: Mean Squared Error for {lang} model: {mean_squared_error(y_train, y_train_pred)}")
        print('='*24)
        # Evaluate the model
        y_pred = mlp.predict(X_test)
        print(y_pred)
        print(y_test)
        
        mse = mean_squared_error(y_test, y_pred)
        # save model
        joblib.dump(mlp, f'models/{lang}_model.pkl')

        print(f"Mean Squared Error for {lang} model: {mse}")
        # compare with baseline
        baseline = np.mean(y_train, axis=0)
        baseline_mse = mean_squared_error(y_test, np.full_like(y_test, baseline))
        print(f"Mean Squared Error for baseline model: {baseline_mse}")

if __name__ == '__main__':
    main()