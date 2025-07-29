# 🎬 LLM Powered Movie Recommender System (IMDB Top 1000)

This project builds a **movie recommendation system** that combines **semantic similarity search**, **emotion classification**, and **data visualization** using state-of-the-art tools like Hugging Face Transformers, PEFT (LoRA), and LangChain with ChromaDB.

The system takes a movie dataset (IMDB Top 1000 movies), cleans and processes it, performs **emotion detection on movie descriptions**, creates **vector embeddings for semantic search**, and enables querying for movie recommendations based on natural language.

---

## 🚀 Features

- ✅ **Dataset Preparation:** IMDB Top 1000 movies cleaned and processed for modeling.  
- ✅ **Exploratory Data Analysis:** Missing value heatmaps, overview length distribution.  
- ✅ **Emotion Classification:** Fine-tuned **RoBERTa** with **LoRA (PEFT)** on `dair-ai/emotion` dataset.  
- ✅ **Semantic Search Engine:** TF-IDF embeddings stored in **ChromaDB** with metadata linking to movies.  
- ✅ **Query-Based Recommendations:** Retrieve top relevant movies by description similarity.  
- ✅ **Emotion-Augmented Recommendations:** Append emotion labels and confidence scores for each movie.  
- ✅ **Visualization:** Distribution of movie overview lengths, emotion predictions, and genre counts.  
- ✅ **Model Export:** Trained model is saved and downloadable as a `.zip`.

---

## 🛠️ Tools & Libraries

### Core Libraries
- [Transformers](https://huggingface.co/transformers/): RoBERTa for emotion classification  
- [PEFT](https://github.com/huggingface/peft): LoRA for parameter-efficient fine-tuning  
- [Datasets](https://huggingface.co/docs/datasets): Loading `dair-ai/emotion` dataset  
- [LangChain](https://www.langchain.com/): For vector database integration and semantic search  
- [Chroma](https://docs.trychroma.com/): Persistent vector store for fast similarity search  
- [TF-IDF](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html): Embedding movie descriptions for semantic retrieval  

### Data Handling
- **pandas** – Data cleaning and manipulation  
- **numpy** – Numerical operations  
- **matplotlib & seaborn** – Data visualization  

### Gradio Dashboard

![gradio_dashboard](https://github.com/user-attachments/assets/dbd7f82c-3478-444f-9aee-73ed16ee1b65)



