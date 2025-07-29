import pandas as pd
import numpy as np
import gradio as gr
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain.embeddings.base import Embeddings
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List
import re

# Load your movie dataset
movies = pd.read_csv("movies_final.csv")

# Custom TF-IDF Embeddings class
class TFIDFEmbeddings(Embeddings):
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.fitted = False
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        self.fitted = True
        embeddings = []
        for i in range(tfidf_matrix.shape[0]):
            embedding = tfidf_matrix[i].toarray()[0].tolist()
            embeddings.append(embedding)
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        if not self.fitted:
            raise ValueError("Call embed_documents first!")
        query_vector = self.vectorizer.transform([text])
        return query_vector.toarray()[0].tolist()

print("Loading and processing movie descriptions...")

# Load tagged descriptions and create documents
try:
    raw_documents = TextLoader("tagged_description.txt").load()
    raw_content = raw_documents[0].page_content
    lines = raw_content.strip().split('\n')
    
    documents_with_metadata = []
    
    for line_num, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue
        
        patterns = [
            r'^["\']?(\d+)["\']?\s+(.+)$',
            r'^(\d+)\s*[\-:]\s*(.+)$',
            r'^(\d+)\s+(.+)$',
        ]
        
        extracted = False
        for pattern in patterns:
            match = re.match(pattern, line)
            if match:
                try:
                    movie_index = int(match.group(1))
                    movie_content = match.group(2).strip()
                    
                    if movie_content:
                        doc = Document(
                            page_content=movie_content,
                            metadata={'movie_index': movie_index, 'source': 'tagged_description.txt'}
                        )
                        documents_with_metadata.append(doc)
                        extracted = True
                        break
                except ValueError:
                    continue
        
        if not extracted:
            print(f"Failed to parse line {line_num}: {line[:50]}...")
    
    print(f"Successfully processed {len(documents_with_metadata)} movie descriptions")
    
    embeddings = TFIDFEmbeddings()
    vectordb = Chroma.from_documents(documents=documents_with_metadata, embedding=embeddings)
    print("✅ Vector database created successfully!")
    
    vector_search_available = True
    
except FileNotFoundError:
    print("❌ tagged_description.txt not found. Vector search will be disabled.")
    vector_search_available = False
    vectordb = None

def recommend_movies(query: str, selected_genre: str, selected_emotion: str):
    if not query.strip():
        return []
    
    try:
        if not vector_search_available:
            movies_filtered = movies.copy()
            query_lower = query.lower()
            mask = movies_filtered['Overview'].str.lower().str.contains(query_lower, na=False)
            movies_filtered = movies_filtered[mask]
        else:
            docs = vectordb.similarity_search(query, k=50)
            recommended_indexes = [doc.metadata['movie_index'] for doc in docs]
            movies_filtered = movies[movies.index.isin(recommended_indexes)]
        
        # ✅ Filter by genre
        if selected_genre != "All":
            movies_filtered = movies_filtered[
                movies_filtered['Predicted_Genres_Top3']
                .fillna("")
                .str.contains(selected_genre, case=False, na=False)
            ]
        
        # ✅ Filter by emotion
        if selected_emotion != "All":
            emotion_mapping = {
                "Joy": "joy",
                "Sadness": "sadness",
                "Anger": "anger",
                "Fear": "fear",
                "Surprise": "surprise"
            }
            if selected_emotion in emotion_mapping:
                emotion_col = "emotion"
                confidence_col = "emotion_confidence"
                
                movies_filtered = movies_filtered[
                    movies_filtered[emotion_col].str.lower() == emotion_mapping[selected_emotion]
                ]
                
                # Sort by confidence
                if confidence_col in movies_filtered.columns:
                    movies_filtered = movies_filtered.sort_values(by=confidence_col, ascending=False)
        
        recommendations = movies_filtered.head(16)
        
        results = []
        for _, row in recommendations.iterrows():
            title = row.get("Series_Title", "Unknown Title")
            overview = row.get("Overview", "No description available.")
            
            if pd.notna(overview):
                truncated_desc_split = str(overview).split()
                truncated_description = " ".join(truncated_desc_split[:30]) + "..."
            else:
                truncated_description = "No description available."
            
            year = ""
            if "Released_Year" in row and pd.notna(row["Released_Year"]):
                try:
                    year = f" ({str(row['Released_Year'])})"
                except:
                    year = ""
            
            rating = ""
            if "IMDB_Rating" in row and pd.notna(row["IMDB_Rating"]):
                rating = f" | ⭐ {row['IMDB_Rating']:.1f}"
            
            genres = ""
            if pd.notna(row.get("Predicted_Genres_Top3", "")):
                genres = f" | {row['Predicted_Genres_Top3']}"
            
            caption = f"{title}{year}{rating}\n{genres}\n{truncated_description}"
            
            poster_url = row.get("Poster_Link", "https://via.placeholder.com/300x450/cccccc/666666?text=No+Poster")
            if pd.isna(poster_url) or not poster_url:
                poster_url = "https://via.placeholder.com/300x450/cccccc/666666?text=No+Poster"
            
            results.append((poster_url, caption))
        
        return results
        
    except Exception as e:
        print(f"Error in recommend_movies: {e}")
        return []

# Genres and emotions
genres = ["All", "Action", "Adventure", "Animation", "Comedy", "Crime", "Drama", 
          "Family", "Fantasy", "Horror", "Mystery", "Romance", "Sci-Fi", 
          "Thriller", "War", "Western", "Biography", "Documentary", "History"]

emotions = ["All", "Joy", "Sadness", "Anger", "Fear", "Surprise"]

# Gradio Dashboard
with gr.Blocks(theme=gr.themes.Soft(), title="Movie Recommender") as dashboard:
    gr.Markdown("# 🎬 Movie Recommender")
    if vector_search_available:
        gr.Markdown("✅ **LangChain + TF-IDF Vector Search Active** - Enter a movie description and filter by genre and emotion!")
    else:
        gr.Markdown("⚠️ **Fallback Mode** - Vector search unavailable. Using simple text matching.")
    
    with gr.Row():
        with gr.Column(scale=3):
            user_query = gr.Textbox(
                label="Movie Description",
                placeholder="e.g., A movie about adventures and fantasy, action heroes saving the world, romantic story in Paris...",
                lines=3
            )
        
        with gr.Column(scale=1):
            genre_dropdown = gr.Dropdown(
                choices=genres,
                label="Genre Filter",
                value="All"
            )
            
            emotion_dropdown = gr.Dropdown(
                choices=emotions,
                label="Emotion Filter",
                value="All"
            )
    
    submit_button = gr.Button("🔍 Find Similar Movies", variant="primary", size="lg")
    
    gr.Markdown("## 🎬 Movie Recommendations")
    output_gallery = gr.Gallery(
        label="Recommended Movies",
        columns=4,
        rows=4,
        height="auto",
        object_fit="contain"
    )
    
    submit_button.click(
        fn=recommend_movies,
        inputs=[user_query, genre_dropdown, emotion_dropdown],
        outputs=output_gallery
    )
    
    gr.Markdown("""
    ### 💡 Example Descriptions:
    - "A movie about adventures and fantasy"
    - "A thrilling story with heroes saving the world"
    - "A romantic comedy set in a big city"  
    - "Dark psychological thriller with mysterious characters"
    - "Family-friendly animated movie with talking animals"
    - "Sci-fi story about space exploration and alien contact"
    - "A dramatic story about overcoming personal struggles"
    """)
    
    with gr.Accordion("🔧 Technical Details", open=False):
        if vector_search_available:
            gr.Markdown(f"""
            **Vector Search Setup:**
            - Documents processed: {len(documents_with_metadata) if 'documents_with_metadata' in locals() else 'N/A'}
            - Embedding method: TF-IDF (max_features=1000)
            - Vector database: Chroma
            - Search results: Top 50 similar documents
            - Final results: Top 16 after filtering
            """)
        else:
            gr.Markdown("""
            **Fallback Mode:**
            - Using simple text matching in movie overviews
            - Requires: tagged_description.txt file
            """)

if __name__ == "__main__":
    print("Starting Movie Recommender Dashboard...")
    print(f"Loaded {len(movies)} movies from movie_final.csv")
    print("Available genres:", genres[1:])
    print("Available emotions:", emotions[1:])
    
    if vector_search_available:
        print("✅ Vector search is active")
        try:
            test_query = "A movie about adventures and fantasy"
            test_docs = vectordb.similarity_search(test_query, k=5)
            print(f"✅ Test search successful: Found {len(test_docs)} documents for '{test_query}'")
        except Exception as e:
            print(f"❌ Test search failed: {e}")
    else:
        print("⚠️ Vector search is not available - using fallback mode")
    
    dashboard.launch(
        share=False,
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True,
        inbrowser=True
    )
