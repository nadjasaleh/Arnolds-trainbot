# Group 4 Projekt
# 🏋️ Arnold AI – Personal Trainer Chatbot  

**Project Plan**  

AI chatbot that acts as a motivational personal trainer inspired by Arnold Schwarzenegger.  
Uses RAG (Retrieval-Augmented Generation) with an exercise knowledge base.  

---

# 🎯 Business Goal

Build an engaging AI personal trainer that:

- Creates structured workout plans  
- Grounds recommendations in a curated exercise knowledge base  
- Motivates users with a strong, confident persona as Arnold  
- Handles unsafe inputs responsibly  

Deliverables:
- Architecture overview  
- Business goal, success criteria, risks & constraints  
- Working demo  
- GitHub repository with code  

---

# 👥 Structure

### 🔹RAG & Backend
- Exercise knowledge base (JSON)
- Embeddings
- Retrieval logic
- Context injection into prompt

### 🔹Prompt & Guardrails
- Arnold persona system prompt
- Structured output enforcement
- Safety checks
- Unsafe input handling

### 🔹Frontend
- Streamlit UI
- Input form (goal, level, equipment, injuries)
- Clean output formatting
- Demo scenarios

### 🔹Architecture & Presentation
- Architecture diagram
- Business framing
- Risks & constraints
- Demo coordination

---

# 🗓 Day 1 – Core Build

## ✅ Define Scope 
User inputs:
- Goal
- Experience level
- Equipment
- Days per week
- Injuries
- Dietplan

Bot outputs:
- Personalized workout plan and dietplan
- Exercises (from KB only)
- Ex: Weekly split, Sets & reps, Rest times
- Safety notes
- Funny Arnold quotes

---

## ✅ Build Exercise Knowledge Base
- A couple of exercises
- Categorized by muscle group, equipment, difficulty
- Include safety notes & programming ranges

---

## ✅ Implement RAG
Flow:
User Query → Embed → Retrieve Top Exercises → Inject into Prompt → Generate Plan

Goal:
- No hallucinated exercises
- Relevant exercise retrieval

---

## ✅ Implement Arnold Persona
- Confident, disciplined, motivational tone
- Structured responses
- Focus only on fitness
- No medical advice

End of Day 1:
- Working chatbot
- RAG integrated
- Persona active

---

# 🗓 Day 2 – Polish & Presentation

## ✅ Enforce Structured Output
Response sections:
1. Weekly Plan  
2. Daily Breakdown  
3. Exercises  
4. Sets & Reps  
5. Rest  
6. Safety Notes  
7. Motivational Close  

---

## ✅ Add Guardrails
Handle:
- Extreme training requests
- Injury misuse
- Medical-related prompts


---

## ✅ Architecture Slide
Show:
- Streamlit UI  
- Backend  
- RAG Retriever  
- Exercise KB  
- Azure OpenAI (Embeddings + Chat)  

Simple data flow diagram.

---

## ✅ Prepare Demo
3 scenarios:
1. Beginner muscle gain  
2. Knee pain modification  
3. Unsafe extreme request  

# Notes

### Data Sources

Book source: https://nnty.fun/downloads/books/TAEL/SELF%20IMPROVEMENT/FITNESS/BODYBUILDING/The%20New%20Encyclopedia%20of%20Modern%20Bodybuilding%20-%20Simon%20%26%20Schuster.%20Arnold%20Schwarzenegger%2C%20Bill%20Dobbins%20%281998%29.pdf

Quotes: https://www.brainyquote.com/authors/arnold-schwarzenegger-quotes

Gifs:  https://giphy.com/explore/arnold-schwarzenegger

### Stack

Frontend: Streamlit

Storage: Azure Cosmos DB for embedding vectors
