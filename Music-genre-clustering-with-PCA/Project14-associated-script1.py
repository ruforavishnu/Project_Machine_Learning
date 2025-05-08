import librosa
import numpy as np
import pandas as pd
import os

folder = 'songs/'
features = []
filenames = []

for file in os.listdir(folder):
    if file.endswith('.mp3'):
        path = os.path.join(folder, file)

        try:
            y, sr = librosa.load(path,  duration=30)
            mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)
            chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)
            contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr), axis=1)
            feature_vector = np.concatenate([mfcc, chroma, contrast])
            features.append(feature_vector)
            filenames.append(file)
        
        except Exception as e:
            print(f'Failed to process {file}: {e}')


df = pd.DataFrame(features)
df['filename'] = filenames


import pandas as pd
import plotly.express as px

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

X = df.drop('filename', axis=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X_pca)



# Example data (replace with your actual arrays)
plot_df = pd.DataFrame({
    'PC1': X_pca[:, 0],
    'PC2': X_pca[:, 1],
    'Cluster': labels,
    'Filename': df['filename']
})

fig = px.scatter(
    plot_df,
    x='PC1',
    y='PC2',
    color=plot_df['Cluster'].astype(str),
    hover_name='Filename',
    title='Interactive Music Clusters',
    width=800,
    height=600
)

# Save and open in browser
fig.write_html("music_clusters.html", auto_open=True)
