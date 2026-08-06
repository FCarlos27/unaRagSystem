# Product Requirements Document (PRD): MVP RAG Microservice for Centro Local Sucre

---

## 1. Executive Summary & Objective

[cite_start]Currently, the web platforms of Centro Local Sucre (UNASUCRE and UNASEC) rely on rigid menus and static PDF documents, lacking semantic search capabilities[cite: 262, 263]. [cite_start]This limitation creates an access barrier and cognitive overload for students, triggering severe saturation of human support channels[cite: 264]. 

[cite_start]**Objective:** Design and implement a containerized MVP software solution based on Retrieval-Augmented Generation (RAG) architecture[cite: 266, 288]. [cite_start]This independent microservice will serve as a semantic bridge, translating informal student queries into precise, contextualized responses extracted exclusively from official university documents[cite: 265, 267].

---

## 2. MVP Scope

[cite_start]To ensure a rapid and robust deployment, the MVP will focus strictly on the core retrieval and generation engine[cite: 290].

**In Scope for MVP:**
* [cite_start]Static ingestion of a pre-selected batch of official university PDFs (e.g., current regulations and the academic calendar)[cite: 291].
* [cite_start]Text chunking, embedding generation, and vector database storage[cite: 292].
* [cite_start]A single RESTful API endpoint to receive queries and return answers[cite: 293].
* [cite_start]Strict hallucination control mechanisms (prompt engineering)[cite: 294].
* [cite_start]Docker containerization of the microservice[cite: 295].

**Out of Scope for MVP (V2 Features):**
* [cite_start]User authentication middleware (handling student logins)[cite: 295].
* [cite_start]Automated live-syncing with the university's CMS databases[cite: 296].
* [cite_start]A graphical User Interface (UI) or frontend chatbot window (the MVP is strictly the backend API)[cite: 296].

---

## 3. Functional Requirements (FR)

* [cite_start]**FR1 - Document Vectorization:** The system must process uploaded institutional knowledge (PDFs), generate text embeddings, and store them in a structured vector database[cite: 297].
* [cite_start]**FR2 - Natural Language Processing (NLP):** The microservice must parse incoming queries, applying tokenization techniques to interpret natural language, slang, and specific university acronyms[cite: 298].
* [cite_start]**FR3 - Contextual Retrieval:** Upon receiving a query, the system must search the vector database and retrieve the most semantically relevant text chunks[cite: 299].
* [cite_start]**FR4 - Generative Response & Hallucination Control:** The system must utilize a Large Language Model (LLM) API to formulate a coherent answer based only on the retrieved context, preventing the generation of unverified information[cite: 300].
* [cite_start]**FR5 - Unanswered Query Logging:** The system must log false positives or questions it could not adequately answer to a standard log file or basic database for future analysis[cite: 301].

---

## 4. Non-Functional Requirements (NFR)

* [cite_start]**NFR1 - Containerization:** The technological solution must be packaged as an independent microservice using Docker to isolate the environment and maintain engineering standards[cite: 302].
* [cite_start]**NFR2 - Scalability:** The architecture must allow the vector database and the NLP processing engine to scale independently if query volume increases[cite: 303].
* [cite_start]**NFR3 - Language Support:** The embedding model and LLM must be optimized for the Spanish language, recognizing Venezuelan and local university dialects[cite: 304].

---

## 5. Technical Stack (MVP Recommendation)

| Component | Recommended Technology | Justification |
| :--- | :--- | :--- |
| **Backend Framework** | FastAPI (Python) | [cite_start]Lightweight, asynchronous, and excellent for building RESTful microservices[cite: 306]. |
| **Vector Database** | ChromaDB (Local) | [cite_start]ChromaDB is ideal for a local, containerized MVP without external DB costs[cite: 307]. |
| **Embedding Model** | Nemotron-3-Embed-1B (via NVIDIA API) | Optimized for multilingual/semantic search, supports Spanish, hosted remotely so no local GPU is required. |
| **LLM / AI API** | DeepSeek-R1-Distill-8B (via NVIDIA API) | Cost-effective reasoning model capable of following strict hallucination-control prompts and Spanish generation. |
| **Orchestration** | LangChain | [cite_start]Standard framework for routing RAG pipelines and managing document retrieval[cite: 309]. |
| **PDF Parsing** | PyPDF2 | Lightweight, dependency-free library for extracting text from official university PDFs. |
| **Text Chunking** | RecursiveCharacterTextSplitter (LangChain) | Splits documents into overlapping, semantically coherent chunks ideal for retrieval. |
| **Infrastructure** | Docker | [cite_start]Ensures the microservice is environment-agnostic and easy to deploy[cite: 310]. |

---

## 6. Core API Endpoint Specification

[cite_start]To achieve Goal 3 (Microservice Enablement), the application will expose the following interface[cite: 311]:

* [cite_start]**Endpoint:** `POST /api/v1/query` [cite: 311]
* [cite_start]**Description:** Receives a student's natural language question and returns the RAG-generated answer[cite: 311].
* [cite_start]**Request Payload:** Requires a JSON object containing the user's natural language query[cite: 312].
* [cite_start]**Response Payload:** Returns a JSON object containing the generated answer, confidence metrics, and source document citations[cite: 312].