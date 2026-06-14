import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D

# Load the dataset
train = pd.read_csv('dataset/train.csv')
answers = pd.read_csv('dataset/train_answers.csv')

# Check the number of rows in both datasets
print(f"Train data shape: {train.shape}")
print(f"Answers data shape: {answers.shape}")

# Aligning train and answers based on the number of samples (keep only matching rows)
if train.shape[0] != answers.shape[0]:
    print(f"Warning: Mismatched sample sizes. Trimming the train dataset to match answers.")
    train = train.head(answers.shape[0])

# Feature labels (adjust to your dataset's feature columns)
features_labels = list(train.columns.values[5:])
X = train[features_labels]
y = answers['male']

# Check the shapes of X and y after alignment
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

# Feature Correlation (highlighting the most important features)
def plot_correlation_matrix(X, y):
    # Compute the correlation matrix
    correlation_matrix = X.corr()

    # Correlation with target variable (y)
    correlation_with_target = X.apply(lambda x: x.corr(y))

    # Sort the features by correlation with the target
    sorted_features = correlation_with_target.abs().sort_values(ascending=False)

    # Highlight the top 10 correlated features with the target variable
    top_features = sorted_features.head(10).index.tolist()

    # Plot the correlation matrix for the top features
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix[top_features].loc[top_features], annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
    plt.title('Correlation Heatmap of Important Features (Top 10 with Target)')
    plt.show()

# Visualize the data (input dataset as points on a graph in 3D)
def plot_data_scatter_3d(X, y):
    # Standardize the features (important for PCA and t-SNE)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Reduce the dataset dimensions to 3D for visualization (using PCA for simplicity)
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_scaled)

    # 3D scatter plot (visualize clusters based on gender)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=y, cmap='viridis', alpha=0.7)
    ax.set_title('3D PCA of Input Dataset (Features)')
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')
    plt.legend(*scatter.legend_elements(), title="Gender")
    plt.show()

    # Optionally, you can use t-SNE for better clustering visualization in 3D
    tsne = TSNE(n_components=3, random_state=42)
    X_tsne = tsne.fit_transform(X_scaled)

    # 3D scatter plot for t-SNE
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter_tsne = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], X_tsne[:, 2], c=y, cmap='viridis', alpha=0.7)
    ax.set_title('3D t-SNE of Input Dataset (Features)')
    ax.set_xlabel('t-SNE Component 1')
    ax.set_ylabel('t-SNE Component 2')
    ax.set_zlabel('t-SNE Component 3')
    plt.legend(*scatter_tsne.legend_elements(), title="Gender")
    plt.show()

# Run Visualizations
if __name__ == '__main__':
    print("Visualizing Data...")

    # Feature Correlation (focus on important features)
    #plot_correlation_matrix(X, y)

    # Visualize the data (3D scatter plot with PCA)
    plot_data_scatter_3d(X, y)
