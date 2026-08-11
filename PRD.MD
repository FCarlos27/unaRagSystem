# Product Requirements Document (PRD): MVP RAG Microservice for Centro Local Sucre

---

## 1. Executive Summary & Objective

Currently, the web platforms of Centro Local Sucre (UNASUCRE and UNASEC) rely on rigid menus and static PDF documents, lacking semantic search capabilities. This limitation creates an access barrier and cognitive overload for students, triggering severe saturation of human support channels. 

**Objective:** Design and implement a containerized MVP software solution based on Retrieval-Augmented Generation (RAG) architecture. This independent microservice will serve as a semantic bridge, translating informal student queries into precise, contextualized responses extracted exclusively from official university documents.

---

## 2. MVP Scope

To ensure a rapid and robust deployment, the MVP will focus strictly on the core retrieval and generation engine.

**In Scope for MVP:**
* Static ingestion of a pre-selected batch of official university PDFs (e.g., current regulations and the academic calendar).
* Text chunking, embedding generation, and vector database storage.
* A single RESTful API endpoint to receive queries and return answers.
* Strict hallucination control mechanisms (prompt engineering).
* Docker containerization of the microservice.

**Out of Scope for MVP (V2 Features):**
* User authentication middleware (handling student logins).
* Automated live-syncing with the university's CMS databases.
* A graphical User Interface (UI) or frontend chatbot window (the MVP is strictly the backend API).

---

## 3. Functional Requirements (FR)

* **FR1 - Document Vectorization:** The system must process uploaded institutional knowledge (PDFs), generate text embeddings, and store them in a structured vector database.
* **FR2 - Natural Language Processing (NLP):** The microservice must parse incoming queries, applying tokenization techniques to interpret natural language, slang, and specific university acronyms.
* **FR3 - Contextual Retrieval:** Upon receiving a query, the system must search the vector database and retrieve the most semantically relevant text chunks.
* **FR4 - Generative Response & Hallucination Control:** The system must utilize a Large Language Model (LLM) API to formulate a coherent answer based only on the retrieved context, preventing the generation of unverified information.
* **FR5 - Unanswered Query Logging:** The system must log false positives or questions it could not adequately answer to a standard log file or basic database for future analysis.

---

## 4. Non-Functional Requirements (NFR)

* **NFR1 - Containerization:** The technological solution must be packaged as an independent microservice using Docker to isolate the environment and maintain engineering standards.
* **NFR2 - Scalability:** The architecture must allow the vector database and the NLP processing engine to scale independently if query volume increases.
* **NFR3 - Language Support:** The embedding model and LLM must be optimized for the Spanish language, recognizing Venezuelan and local university dialects.

---

## 5. Technical Stack (MVP Recommendation)

| Component | Recommended Technology | Justification |
| :--- | :--- | :--- |
| **Backend Framework** | FastAPI (Python) | Lightweight, asynchronous, and excellent for building RESTful microservices. |
| **Vector Database** | ChromaDB (Local) | ChromaDB is ideal for a local, containerized MVP without external DB costs. |
| **Embedding Model** | nomic-embed-text (local via Ollama) | Lightweight local embedding model (274 MB) with strong multilingual/semantic search performance and no external API costs. |
| **LLM / AI API** | Llama-3.2-3B (local via Ollama) | Lightweight local generative model capable of following strict hallucination-control prompts and Spanish generation, with no external API costs. |
| **Orchestration** | LangChain | Standard framework for routing RAG pipelines and managing document retrieval. |
| **PDF Parsing** | PyPDF2 | Lightweight, dependency-free library for extracting text from official university PDFs. |
| **Text Chunking** | RecursiveCharacterTextSplitter (LangChain) | Splits documents into overlapping, semantically coherent chunks ideal for retrieval. |
| **Infrastructure** | Docker | Ensures the microservice is environment-agnostic and easy to deploy. |

---

## 6. Core API Endpoint Specification

To achieve Goal 3 (Microservice Enablement), the application will expose the following interface:

* **Endpoint:** `POST /api/v1/query` 
* **Description:** Receives a student's natural language question and returns the RAG-generated answer.
* **Request Payload:** Requires a JSON object containing the user's natural language query.
* **Response Payload:** Returns a JSON object containing the generated answer, confidence metrics, and source document citations.