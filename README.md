# 🎬 LLM-Powered Movie Recommender System (IMDB Top 1000)

This project is an **AI-powered movie recommendation system** that combines:
- **Similarity search** using TF-IDF embeddings and Chroma vector database.
- **Zero-shot genre classification** using a Hugging Face model.
- **Emotion classification** using a fine-tuned model on the `dair-ai/emotion` dataset.
- An **interactive Gradio dashboard** for movie search and filtering.

The system enables users to search for movies by description, retrieve similar movies, and filter them based on predicted emotions.

---

## 📂 Dataset
We used the **[IMDB Top 1000 Movies Dataset](https://www.kaggle.com/datasets/PromptCloudHQ/imdb-data)** obtained from Kaggle.  
The dataset was cleaned, and additional columns were added for predicted **emotions** and **genres**.

---

## 🧠 Models

### 1. Zero-Shot Genre Classification
- **Model**: [facebook/bart-large-mnli](https://huggingface.co/facebook/bart-large-mnli)  
- **Purpose**: Classify each movie into one or more of the following **18 genres** based on the movie overview:

- **Technique**: Zero-shot text classification using Hugging Face's `transformers` library.

---

### 2. Emotion Classification
- **Dataset**: [dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion)
- **Model**: [roberta-base](https://huggingface.co/roberta-base) fine-tuned on the dataset.
- **Emotions**:  

- **Dataset split**:
- Train: 16k samples
- Validation: 2k samples
- Test: 2k samples
- **Accuracy**: ~50% (due to dataset imbalance).

This model predicts the **dominant emotion** of a movie based on its overview.

---

## 🏗️ Pipeline

1. **Data Cleaning**  
 - Loaded the Kaggle dataset using Pandas.
 - Removed duplicates and null values.
 - Standardized column names.

2. **Genre Classification**  
 - Used `facebook/bart-large-mnli` for zero-shot genre classification.
 - Added predicted genres as a new column in the DataFrame.

3. **Emotion Classification**  
 - Fine-tuned `roberta-base` on `dair-ai/emotion`.
 - Added predicted emotion and confidence scores as new columns.

4. **Vector Database Creation**  
 - Computed TF-IDF embeddings for movie overviews.
 - Stored embeddings in **Chroma** for similarity search.

5. **Final Dataset**  
 - Merged the predictions (emotions + genres) into the original dataset.
 - Exported the final enriched dataset as `movies_final.csv`.

6. **Gradio Dashboard**  
 - Built an interactive dashboard using [Gradio](https://www.gradio.app/).
 - Features:
   - Search for similar movies using movie descriptions.
   - Filter movies based on predicted emotions.
   - Display results in a clean and interactive interface.

---

## 🛠️ Tech Stack

| Component            | Technology |
|----------------------|------------|
| Programming Language | Python     |
| Libraries            | Pandas, NumPy, Scikit-learn, Hugging Face Transformers, LangChain, ChromaDB, Gradio |
| Dataset              | [IMDB Top 1000 Movies](https://www.kaggle.com/datasets/PromptCloudHQ/imdb-data) |
| Models               | [facebook/bart-large-mnli](https://huggingface.co/facebook/bart-large-mnli), [roberta-base](https://huggingface.co/roberta-base) |

---

## 🚀 How to Run

1. **Clone the repository**
 ```bash
 git clone https://github.com/your-username/movie-recommender.git
 cd movie-recommender
