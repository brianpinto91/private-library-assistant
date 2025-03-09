# private-library-assistant
A RAG based LLM agent that retrieves information from your private collection of digital books.

### Objective:
This project builds an **AI-powered question-answering system** that enables users to query a large collection of private documents and books and receive contextually relevant answers. It leverages **Retrieval-Augmented Generation (RAG)** to improve response accuracy by retrieving relevant text from the stored documents in your private collection before generating an answer.

### 🔍 The Problem It Solves
1. **No information accessibilty**
   - Traditional internet search methods do not have acces to your private documents or contents of the purchased books (legally).
   - The ChatGPT relies on open source information or summaries of paid books. So as an user you will get summary of summaries, which may or might not be accurate depending on the sources.

2. **Difficulty in Finding Information in Books**  
   - Users often struggle to search for specific information across multiple books.  
   - Users might want to know where exactly certain information is in the books.

3. **Challenges with AI Hallucination**  
   - Large Language Models (LLMs) often generate **incorrect or unverifiable** answers.  
   - Without access to real documents, AI might **make up information** instead of relying on actual content.


### 💡 How This Project Solves It
1. **Extracts & Processes Books Efficiently**  
   - Converts **PDF books** into **clean, structured text**.  
   - Splits content into **meaningful chunks** for efficient retrieval.  

2. **Stores Chunks & Embeddings in SQLite**  
   - Uses **SQLite database** to persist book chunks and their embeddings.  
   - Makes querying fast and memory-efficient.  

3. **Retrieves the Most Relevant Context Using Vector Search**  
   - Uses **FAISS (Facebook AI Similarity Search)** for fast semantic retrieval.  
   - Retrieves the **most relevant book passages** before answering queries.  

4. **Generates Answers Using an LLM**  
   - Passes the retrieved content to an **LLM** to generate **accurate, context-based responses**.  
   - Ensures that answers are grounded in **real book content**, reducing hallucinations.  
   - Gives references like page number and the book title for user to verfiy the answer or find more information.
