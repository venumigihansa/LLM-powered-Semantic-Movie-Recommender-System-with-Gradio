🎬 LLM Powered Movie Recommender System (IMDB Top 1000)
This project builds a movie recommendation system that combines semantic similarity search, genre classification, and emotion classification using modern NLP frameworks like Hugging Face Transformers and LangChain with ChromaDB.

The system takes the IMDB Top 1000 Movies Dataset from Kaggle, cleans and processes it, performs zero-shot genre classification and emotion detection, generates vector embeddings for semantic search, and allows users to retrieve movie recommendations based on natural language queries.

🚀 Features
✅ Dataset Preparation: Cleaned and preprocessed IMDB Top 1000 movies dataset from Kaggle.

✅ Zero-Shot Genre Classification: Used facebook/bart-large-mnli to classify movies into 18 genres:
Action, Adventure, Animation, Comedy, Crime, Drama, Family, Fantasy, Horror, Mystery, Romance, Sci-Fi, Thriller, War, Western, Biography, Documentary, History

✅ Emotion Classification: Fine-tuned roberta-base on dair-ai/emotion to classify movies into 6 emotions:
Sadness, Joy, Love, Anger, Fear, Surprise

✅ Similarity Search Engine: Built semantic search using TF-IDF embeddings stored in ChromaDB with LangChain integration.

✅ Emotion-Augmented Recommendations: Appended emotion labels and confidence scores to the movie dataset.

✅ Final Consolidated Dataset: Created movies_final.csv with cleaned data, predicted genres, and emotion scores.

✅ Interactive Dashboard: Implemented a Gradio interface where users can:

Search movies based on natural language descriptions.

Filter results by emotion labels.

View confidence scores for predictions.

✅ Data Visualization: Plotted emotion distributions, genre counts, and overview statistics.

🛠️ Tools & Libraries
Core Libraries
Transformers: Zero-shot classification and fine-tuning for emotion detection

LangChain: ChromaDB integration for semantic similarity search

ChromaDB: Vector database for movie embeddings

Pandas: Data cleaning and manipulation

NumPy: Numerical processing for dataset transformations

Gradio: Interactive web-based dashboard for movie search

📊 Emotion Model Training
Dataset: dair-ai/emotion

Train/Val/Test Split: 16k / 2k / 2k

Emotions: Sadness, Joy, Love, Anger, Fear, Surprise

Model: Fine-tuned roberta-base

Achieved Accuracy: ~50% (due to class imbalance in the dataset)

🏗 Workflow
Data Cleaning: Removed missing values and standardized dataset structure.

Emotion Classification: Fine-tuned RoBERTa and predicted emotions with confidence scores.

Genre Classification: Applied zero-shot classification using BART-Large-MNLI.

Vector Database Creation: Generated TF-IDF embeddings and stored them in ChromaDB.

Data Consolidation: Created a single movies_final.csv containing cleaned data, genres, and emotions.

Dashboard Development: Built a Gradio-based interface for semantic search and emotion filtering.

🎨 Dashboard Demo
The Gradio dashboard enables:

🔍 Description-Based Search: Retrieve similar movies based on user input.

🎭 Emotion Filtering: Filter movies by emotion predictions.

📊 Metadata Display: View genres, emotions, and confidence scores.

🎥 Demo Video: (Insert demo video link here)
