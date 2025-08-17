import streamlit as st
from openai import OpenAI
import pandas as pd
import json
from datetime import datetime
import hashlib
import numpy as np
import pickle
import os
from sklearn.metrics.pairwise import cosine_similarity
import uuid

# Configure the page
st.set_page_config(
    page_title="Housing Request Classification System",
    page_icon="🏠",
    layout="wide"
)

# Title and description
st.title("🏠 Housing Request Classification System")
st.write("Automated classification with vector-based case similarity matching and AI reasoning.")

# Sidebar for configuration
st.sidebar.header("🔧 Configuration")
openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")

# File paths for vector storage
CASES_FILE = "housing_cases.json"
EMBEDDINGS_FILE = "case_embeddings.pkl"
METADATA_FILE = "system_metadata.json"

# Predefined categories and their descriptions
CATEGORIES = {
    "enquiry": {
        "name": "General Enquiry",
        "description": "Questions about housing policies, procedures, eligibility criteria, or general information requests",
        "keywords": ["question", "inquiry", "information", "how to", "eligibility", "criteria", "policy"],
        "examples": [
            "What are the eligibility criteria for public housing?",
            "How do I apply for housing grants?",
            "What documents do I need for housing application?"
        ]
    },
    "refund": {
        "name": "Refund Request", 
        "description": "Requests for financial refunds, deposit returns, or monetary compensation",
        "keywords": ["refund", "return", "money back", "deposit", "compensation", "reimbursement"],
        "examples": [
            "I need a refund for my housing deposit",
            "Can I get my money back for the cancelled application?",
            "How do I claim compensation for housing delays?"
        ]
    },
    "matrimonial_asset": {
        "name": "Matrimonial Asset Division",
        "description": "Cases involving property division due to divorce, separation, or matrimonial disputes",
        "keywords": ["divorce", "separation", "matrimonial", "spouse", "division", "property split", "ex-husband", "ex-wife"],
        "examples": [
            "I need to transfer my flat ownership after divorce",
            "How to divide property with my ex-spouse?",
            "What are the procedures for matrimonial property settlement?"
        ]
    }
}

class VectorCaseStorage:
    def __init__(self, client):
        self.client = client
        self.cases = self.load_cases()
        self.embeddings = self.load_embeddings()
        self.metadata = self.load_metadata()
    
    def load_cases(self):
        """Load cases from JSON file"""
        if os.path.exists(CASES_FILE):
            with open(CASES_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []
    
    def save_cases(self):
        """Save cases to JSON file"""
        with open(CASES_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.cases, f, indent=2, ensure_ascii=False)
    
    def load_embeddings(self):
        """Load embeddings from pickle file"""
        if os.path.exists(EMBEDDINGS_FILE):
            with open(EMBEDDINGS_FILE, 'rb') as f:
                return pickle.load(f)
        return []
    
    def save_embeddings(self):
        """Save embeddings to pickle file"""
        with open(EMBEDDINGS_FILE, 'wb') as f:
            pickle.dump(self.embeddings, f)
    
    def load_metadata(self):
        """Load system metadata"""
        if os.path.exists(METADATA_FILE):
            with open(METADATA_FILE, 'r') as f:
                return json.load(f)
        return {
            "total_cases": 0,
            "category_counts": {},
            "last_updated": None,
            "system_version": "1.0"
        }
    
    def save_metadata(self):
        """Save system metadata"""
        with open(METADATA_FILE, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def get_embedding(self, text):
        """Get embedding for text using OpenAI"""
        try:
            response = self.client.embeddings.create(
                model="text-embedding-ada-002",
                input=text.strip()
            )
            return response.data[0].embedding
        except Exception as e:
            st.error(f"Error getting embedding: {e}")
            return None
    
    def add_case(self, request_text, classification_result, officer_review=None):
        """Add new case to vector storage"""
        case_id = str(uuid.uuid4())
        case_hash = hashlib.md5(request_text.encode()).hexdigest()
        
        # Check if case already exists
        existing_case = next((case for case in self.cases if case.get('case_hash') == case_hash), None)
        if existing_case:
            return False, "Case already exists"
        
        # Get embedding for the request text
        embedding = self.get_embedding(request_text)
        if embedding is None:
            return False, "Failed to generate embedding"
        
        # Create case record
        case = {
            "id": case_id,
            "case_hash": case_hash,
            "request_text": request_text,
            "predicted_category": classification_result['predicted_category'],
            "confidence_score": classification_result['confidence_score'],
            "reasoning": classification_result['reasoning'],
            "key_indicators": classification_result.get('key_indicators', []),
            "timestamp": datetime.now().isoformat(),
            "officer_review": officer_review,
            "final_category": None
        }
        
        # Add to storage
        self.cases.append(case)
        self.embeddings.append({
            "id": case_id,
            "embedding": embedding,
            "text": request_text
        })
        
        # Update metadata
        self.metadata["total_cases"] = len(self.cases)
        self.metadata["last_updated"] = datetime.now().isoformat()
        category = classification_result['predicted_category']
        self.metadata["category_counts"][category] = self.metadata["category_counts"].get(category, 0) + 1
        
        # Save to files
        self.save_cases()
        self.save_embeddings()
        self.save_metadata()
        
        return True, "Case added successfully"
    
    def find_similar_cases(self, query_text, top_k=5, similarity_threshold=0.7):
        """Find similar cases using vector similarity"""
        if not self.embeddings:
            return []
        
        # Get embedding for query
        query_embedding = self.get_embedding(query_text)
        if query_embedding is None:
            return []
        
        # Calculate similarities
        similarities = []
        for emb_data in self.embeddings:
            similarity = cosine_similarity(
                [query_embedding], 
                [emb_data['embedding']]
            )[0][0]
            
            if similarity >= similarity_threshold:
                # Find corresponding case
                case = next((case for case in self.cases if case['id'] == emb_data['id']), None)
                if case:
                    similarities.append({
                        **case,
                        'similarity_score': similarity
                    })
        
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x['similarity_score'], reverse=True)
        return similarities[:top_k]
    
    def get_statistics(self):
        """Get system statistics"""
        return {
            "total_cases": len(self.cases),
            "category_distribution": self.metadata.get("category_counts", {}),
            "recent_cases": sorted(self.cases, key=lambda x: x['timestamp'], reverse=True)[:5],
            "last_updated": self.metadata.get("last_updated"),
            "average_confidence": np.mean([case['confidence_score'] for case in self.cases]) if self.cases else 0
        }

def create_classification_prompt(request_text, similar_cases_context=""):
    """Create classification prompt with similar cases context"""
    prompt = f"""
You are an AI assistant helping housing officers classify incoming requests with high accuracy.

CLASSIFICATION CATEGORIES:
1. "enquiry" - General questions about housing policies, procedures, eligibility criteria, or information requests
2. "refund" - Requests for financial refunds, deposit returns, monetary compensation, or payment-related issues  
3. "matrimonial_asset" - Property division, ownership transfer, or housing matters related to divorce/separation

CURRENT REQUEST TO CLASSIFY:
{request_text}

{similar_cases_context}

INSTRUCTIONS:
- Analyze the request carefully for intent and key themes
- Consider the similar cases as reference for consistency
- Provide confidence score between 0.0-1.0 (be conservative, use >0.8 only for very clear cases)
- Give detailed reasoning that an officer can understand and verify

RESPONSE FORMAT (valid JSON only):
{{
    "predicted_category": "category_name",
    "confidence_score": 0.85,
    "reasoning": "Clear explanation of classification decision with specific references to key phrases",
    "key_indicators": ["phrase1", "phrase2", "phrase3"],
    "alternative_consideration": "Brief note if other categories were considered"
}}
"""
    return prompt

def initialize_sample_data(vector_storage):
    """Initialize with some sample cases if storage is empty"""
    if len(vector_storage.cases) == 0:
        sample_cases = [
            {
                "text": "I would like to know the eligibility criteria for applying for public housing and what documents I need to submit",
                "category": "enquiry",
                "reasoning": "Clear information request about housing application process"
            },
            {
                "text": "I need a refund for my housing deposit as my application was rejected due to income changes",
                "category": "refund", 
                "reasoning": "Explicit request for financial refund of deposit"
            },
            {
                "text": "My divorce was finalized and I need to transfer the flat ownership to my ex-spouse as per court order",
                "category": "matrimonial_asset",
                "reasoning": "Property transfer related to divorce proceedings"
            },
            {
                "text": "Can I get my money back for the processing fees? My application took too long and I found alternative housing",
                "category": "refund",
                "reasoning": "Request for fee refund due to service delay"
            },
            {
                "text": "What are the procedures for dividing our matrimonial flat after separation?",
                "category": "matrimonial_asset", 
                "reasoning": "Question about property division process after separation"
            }
        ]
        
        for sample in sample_cases:
            result = {
                "predicted_category": sample["category"],
                "confidence_score": 0.95,
                "reasoning": sample["reasoning"],
                "key_indicators": sample["text"].split()[:3]
            }
            vector_storage.add_case(sample["text"], result)

# Main application
if not openai_api_key:
    st.info("Please add your OpenAI API key in the sidebar to continue.", icon="🗝️")
else:
    client = OpenAI(api_key=openai_api_key)
    
    # Initialize vector storage
    if 'vector_storage' not in st.session_state:
        st.session_state.vector_storage = VectorCaseStorage(client)
        # Initialize with sample data if empty
        initialize_sample_data(st.session_state.vector_storage)
    
    vector_storage = st.session_state.vector_storage
    
    # Create layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📋 Request Classification")
        
        # Text input for housing request
        request_text = st.text_area(
            "Enter housing request to classify:",
            height=150,
            placeholder="Example: I would like to know about the refund process for my housing deposit after my divorce settlement..."
        )
        
        # Classification controls
        col_btn, col_sim = st.columns([1, 1])
        with col_btn:
            classify_btn = st.button("🔍 Classify Request", type="primary")
        with col_sim:
            similarity_threshold = st.slider("Similarity Threshold", 0.5, 0.9, 0.7, 0.05)
        
        if classify_btn and request_text.strip():
            with st.spinner("Analyzing request and finding similar cases..."):
                try:
                    # Find similar cases using vector similarity
                    similar_cases = vector_storage.find_similar_cases(
                        request_text, 
                        top_k=3, 
                        similarity_threshold=similarity_threshold
                    )
                    
                    # Build context from similar cases
                    similar_cases_context = ""
                    if similar_cases:
                        similar_cases_context = "\nSIMILAR CASES FOR REFERENCE:\n"
                        for i, case in enumerate(similar_cases, 1):
                            similar_cases_context += f"{i}. [{case['predicted_category'].upper()}] {case['request_text'][:150]}... (Similarity: {case['similarity_score']:.3f})\n"
                        similar_cases_context += "\nUse these cases as reference for consistent classification.\n"
                    
                    # Create classification prompt
                    prompt = create_classification_prompt(request_text, similar_cases_context)
                    
                    # Get AI classification
                    response = client.chat.completions.create(
                        model="gpt-4",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.1
                    )
                    
                    # Parse JSON response
                    result = json.loads(response.choices[0].message.content)
                    
                    # Store in vector database
                    success, message = vector_storage.add_case(request_text, result)
                    
                    if success:
                        st.success("✅ Classification Complete & Stored")
                    else:
                        st.warning(f"⚠️ Classification complete but storage issue: {message}")
                    
                    # Display results
                    category_info = CATEGORIES.get(result['predicted_category'], {})
                    
                    st.subheader("🎯 Classification Result")
                    
                    # Main result display
                    result_col1, result_col2 = st.columns([2, 1])
                    with result_col1:
                        st.success(f"**Category:** {category_info.get('name', result['predicted_category'])}")
                        st.info(f"**Description:** {category_info.get('description', 'N/A')}")
                    
                    with result_col2:
                        confidence = result['confidence_score']
                        if confidence > 0.8:
                            st.metric("Confidence", f"{confidence:.1%}", delta="High")
                        elif confidence > 0.6:
                            st.metric("Confidence", f"{confidence:.1%}", delta="Medium")
                        else:
                            st.metric("Confidence", f"{confidence:.1%}", delta="Low")
                    
                    # Reasoning section
                    st.subheader("💡 Classification Reasoning")
                    st.info(result['reasoning'])
                    
                    # Key indicators
                    if result.get('key_indicators'):
                        st.subheader("🔍 Key Indicators Detected")
                        indicators_html = " ".join([f'<span style="background-color: #e1f5fe; padding: 2px 6px; border-radius: 3px; margin: 2px;">{indicator}</span>' for indicator in result['key_indicators']])
                        st.markdown(indicators_html, unsafe_allow_html=True)
                    
                    # Similar cases reference
                    if similar_cases:
                        st.subheader("📚 Similar Cases Referenced")
                        for i, case in enumerate(similar_cases, 1):
                            with st.expander(f"Case {i}: {case['predicted_category'].title()} | Similarity: {case['similarity_score']:.1%}"):
                                st.write(f"**Request:** {case['request_text'][:300]}{'...' if len(case['request_text']) > 300 else ''}")
                                st.write(f"**Original Reasoning:** {case['reasoning']}")
                                st.write(f"**Date:** {case['timestamp'][:10]}")
                    
                    # Alternative considerations
                    if result.get('alternative_consideration'):
                        st.subheader("⚖️ Alternative Considerations")
                        st.warning(result['alternative_consideration'])
                    
                    # Officer review section
                    st.subheader("👮‍♂️ Officer Review & Feedback")
                    with st.form("officer_review_form"):
                        review_status = st.selectbox(
                            "Classification Review:",
                            ["Pending Review", "✅ Correct", "❌ Should be Enquiry", 
                             "❌ Should be Refund", "❌ Should be Matrimonial Asset"]
                        )
                        
                        officer_notes = st.text_area("Officer Notes (optional):", height=100)
                        
                        if st.form_submit_button("Submit Review"):
                            st.success("✅ Review submitted! This feedback will improve future classifications.")
                            st.balloons()
                
                except json.JSONDecodeError:
                    st.error("Error parsing AI response. Please try again.")
                except Exception as e:
                    st.error(f"Error during classification: {str(e)}")
        
        elif classify_btn:
            st.warning("Please enter a request to classify.")
    
    with col2:
        st.header("📊 System Dashboard")
        
        # Get current statistics
        stats = vector_storage.get_statistics()
        
        # Key metrics
        st.subheader("📈 Key Metrics")
        col_metric1, col_metric2 = st.columns(2)
        with col_metric1:
            st.metric("Total Cases", stats['total_cases'])
        with col_metric2:
            st.metric("Avg Confidence", f"{stats['average_confidence']:.1%}")
        
        # Category distribution
        st.subheader("📂 Category Distribution")
        if stats['category_distribution']:
            for category, count in stats['category_distribution'].items():
                percentage = (count / stats['total_cases']) * 100 if stats['total_cases'] > 0 else 0
                st.write(f"**{category.title()}:** {count} cases ({percentage:.1f}%)")
        else:
            st.write("No cases processed yet")
        
        # Classification categories info
        st.subheader("📋 Classification Guide")
        for key, category in CATEGORIES.items():
            with st.expander(f"{category['name']}"):
                st.write(category['description'])
                st.write("**Example requests:**")
                for example in category['examples']:
                    st.write(f"• {example}")
        
        # Recent activity
        st.subheader("🕐 Recent Classifications")
        if stats['recent_cases']:
            for i, case in enumerate(stats['recent_cases'][:3]):
                with st.expander(f"Case {i+1}: {case['predicted_category'].title()}"):
                    st.write(f"**Text:** {case['request_text'][:100]}...")
                    st.write(f"**Confidence:** {case['confidence_score']:.1%}")
                    st.write(f"**Date:** {case['timestamp'][:19]}")
        
        # System info
        st.subheader("ℹ️ System Information")
        st.write(f"**Last Updated:** {stats['last_updated'][:19] if stats['last_updated'] else 'Never'}")
        st.write(f"**Storage:** File-based Vector DB")
        st.write(f"**Similarity Model:** text-embedding-ada-002")
        
        # Data management
        st.subheader("🔧 Data Management")
        if st.button("📥 Export Cases"):
            st.download_button(
                "Download Cases JSON",
                json.dumps(vector_storage.cases, indent=2),
                "housing_cases_export.json",
                "application/json"
            )
        
        if st.button("🗑️ Clear All Data", type="secondary"):
            if st.checkbox("I understand this will delete all cases"):
                # Clear files
                for file in [CASES_FILE, EMBEDDINGS_FILE, METADATA_FILE]:
                    if os.path.exists(file):
                        os.remove(file)
                st.session_state.vector_storage = VectorCaseStorage(client)
                st.success("All data cleared!")
                st.rerun()

# Footer
st.markdown("---")
st.markdown("*Vector-based semantic similarity ensures more intelligent case matching and consistent classifications.*")