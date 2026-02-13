# Lab4

CS 5542 Lab 4 — Personal README
Project: RAG Application Integration, Deployment, and Monitoring

Name: Ruixuan Hou
Course: Big Data Analytics and Applications (CS 5542)
Date: February 12, 2026

1. Project Overview

This project extended a prototype Retrieval-Augmented Generation (RAG) pipeline into a deployable application. The system integrates a knowledge graph and LLM-based answer generation. I applied the pipeline to a project-aligned dataset, which included technical papers and structured knowledge relevant to our course project.

Dataset Highlights:

"HyperGraphRAG: Retrieval-Augmented Generation via Hypergraph-Structured Knowledge Representation"

"LLM-Powered Knowledge Graphs for Enterprise Intelligence and Analytics"

The text data was chunked and indexed using a hybrid retrieval approach combining dense embeddings (all-MiniLM-L6-v2) and sparse BM25 search, with re-ranking via a cross-encoder.

2. Components Implemented

Frontend: Streamlit application providing a chat-style interface and retrieval configuration.

Backend: FastAPI service handling the core retrieval and answer generation logic.

Logging/Monitoring: CSV-based automatic logging of queries, metrics (Precision@5, Recall@10), and missing-evidence behavior.

Deployment: Deployed the system on Streamlit Community Cloud, with the frontend automatically starting the FastAPI backend.

3. Local Run Instructions

Install dependencies:

pip install -r requirements.txt

Run the application locally:

streamlit run app/main.py

The frontend automatically launches the FastAPI backend if needed.

4. Personal Notes on Lab Experience

I conducted different experiments in Colab to validate retrieval and answer generation.

The retrieval component was the most time-consuming due to chunking, indexing, and hybrid search tuning.

I implemented TF-IDF retrieval for demonstration and verified that all Mini Gold Set queries returned non-empty results.

The Streamlit UI provided a clear interface for query testing and metric display.

Logging enabled evaluation of missing-evidence responses and faithfulness checks.

Personal Notes on Lab Experience

In Colab, I uploaded project files directly using:

from google.colab import files
uploaded = files.upload()

This allowed me to test different documents and CSVs interactively without downloading locally.

Conducted multiple retrieval experiments to verify TF-IDF indexing, hybrid retrieval, and top-K ranking.

The retrieval component was the most time-consuming due to chunking, indexing, and hybrid search tuning.

Verified that Mini Gold Set queries all returned retrieval hits, confirming correct indexing.

Implemented TF-IDF retrieval as a baseline and tested answer grounding with a simple stub function.

Explored logging and evaluation by tracking metrics like Precision@5, Recall@10, faithfulness, and missing-evidence handling.

Key tricks learned:

Truncating retrieved text chunks for UI display.

Normalizing document/evidence IDs for consistent evaluation.

Using json.dumps to store lists of retrieved IDs in logs.

Rapidly testing different query and answer generation pipelines in Colab before deployment.

5. Evaluation
ID	Query	System Result	Pass/Fail
Q1	Primary limitation of HyperGraphRAG	Correctly identified "binary relations" and "representation sparsity"	PASS
Q2	Smart-Summarizer module processing	Explained entity/relation extraction	PASS
Q3	Graph structure of HyperGraphRAG	Correctly identified "Bipartite Graph Storage"	PASS
Q4	Knowledge representation difference	Described difference between Chunk-based, GraphRAG, and HyperGraphRAG	PASS
Q5	Chemical properties of Hydrogen fuel cells	Correctly responded: "Not enough evidence in the retrieved context."	PASS

All Mini Gold Set queries returned retrieval hits, confirming that the TF-IDF baseline worked for the demo corpus.

6. Skills and Takeaways

Learned to integrate hybrid retrieval with LLM-based answer generation.

Gained experience in Streamlit UI development, FastAPI backend deployment, and logging/monitoring of query metrics.

Improved understanding of system engineering for real-world RAG applications.
