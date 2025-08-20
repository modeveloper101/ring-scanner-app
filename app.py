import open_clip
from PIL import Image
import torch
from torchvision import transforms
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.query import MetadataQuery
import streamlit as st
import os
import time

# Configure page
st.set_page_config(
    page_title="Ring Scanner",
    page_icon="💍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        font-size: 3rem;
        margin-bottom: 2rem;
    }
    .match-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    .metric-container {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    .upload-section {
        border: 2px dashed #cccccc;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_model():
    """Initialize and cache the CLIP model"""
    try:
        model, _, preprocess = open_clip.create_model_and_transforms(
            'ViT-B-32', 
            pretrained='laion2b_s34b_b79k'
        )
        return model, preprocess
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

@st.cache_resource
def initialize_weaviate():
    """Initialize and cache Weaviate connection"""
    try:
        weaviate_url = os.environ.get("WEAVIATE_URL")
        weaviate_api_key = os.environ.get("WEAVIATE_API_KEY")
        
        if not weaviate_url or not weaviate_api_key:
            st.error("⚠️ Weaviate credentials not found in environment variables")
            return None, None
            
        client = weaviate.connect_to_weaviate_cloud(
            cluster_url=weaviate_url,
            auth_credentials=Auth.api_key(weaviate_api_key),
        )
        collection = client.collections.get("ModelsCollection")
        return client, collection
    except Exception as e:
        st.error(f"Error connecting to Weaviate: {str(e)}")
        return None, None

def generate_embedding(image, model, preprocess):
    """Generate image embedding using CLIP model (truncated to 10 dimensions)"""
    try:
        image_input = preprocess(image).unsqueeze(0)
        
        with torch.no_grad():
            image_features = model.encode_image(image_input)
        
        # Truncate to first 10 dimensions to match stored data
        return image_features[0].tolist()[:10]
    except Exception as e:
        st.error(f"Error generating embedding: {str(e)}")
        return None

def search_similar_rings(collection, embedding, limit=3):
    """Search for similar rings in Weaviate"""
    try:
        results = collection.query.near_vector(
            near_vector=embedding,
            limit=limit,
            return_metadata=MetadataQuery(distance=True)
        )
        return results
    except Exception as e:
        st.error(f"Search error: {str(e)}")
        return None

# Main app layout
st.markdown('<h1 class="main-header">💍 Ring Scanner</h1>', unsafe_allow_html=True)

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")
    search_limit = st.slider("Number of matches", min_value=1, max_value=5, value=3)
    show_distance = st.checkbox("Show similarity distance", value=True)
    
    st.header("📊 App Status")
    
    # Initialize components
    model, preprocess = initialize_model()
    client, collection = initialize_weaviate()
    
    if model is not None:
        st.success("✅ CLIP Model loaded")
    else:
        st.error("❌ Model loading failed")
        
    if client is not None and collection is not None:
        st.success("✅ Weaviate connected")
    else:
        st.error("❌ Weaviate connection failed")

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.subheader("📤 Upload Ring Photo")
    uploaded_file = st.file_uploader(
        "Choose a ring image", 
        type=["jpg", "jpeg", "png"],
        help="Upload a clear photo of the ring you want to identify"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(
            image, 
            caption="📷 Uploaded Ring Photo", 
            use_container_width=True,
            width=300
        )
        
        # Image info
        st.info(f"📏 Image size: {image.size[0]}x{image.size[1]} pixels")

with col2:
    if uploaded_file and model is not None and collection is not None:
        st.subheader("🔍 Search Results")
        
        # Show loading spinner
        with st.spinner("🔄 Analyzing image and searching..."):
            # Generate embedding
            embedding = generate_embedding(image, model, preprocess)
            
            if embedding:
                # st.success(f"✅ Generated 10-dimensional embedding (truncated from 512)")
                
                # Search for matches
                results = search_similar_rings(collection, embedding, search_limit)
                
                if results and results.objects:
                    # st.success(f"🎯 Found {len(results.objects)} match(es)")
                    
                    # Display matches
                    for i, obj in enumerate(results.objects, start=1):
                        with st.container():
                            # st.markdown(f"""
                            # <div class="match-card">
                            #     <h3>🏆 Match #{i}</h3>
                            # </div>
                            # """, unsafe_allow_html=True)
                            
                            # Create metrics layout
                            metric_col1, metric_col2 = st.columns(2)
                            
                            with metric_col1:
                                st.metric("Name", obj.properties.get("name", "N/A"))
                                st.metric("Price", obj.properties.get("price", "N/A"))
                            
                            with metric_col2:
                                st.metric("Object ID", obj.properties.get("ringId", obj.properties.get("ringId", "N/A")))
                                if show_distance and hasattr(obj, 'metadata') and obj.metadata.distance is not None:
                                    similarity_score = max(0, 1 - obj.metadata.distance)
                                    st.metric("Similarity", f"{similarity_score:.2%}")
                            
                            # Description
                            description = obj.properties.get("description", "No description available")
                            st.text_area(
                                f"Description (Match #{i})", 
                                value=description, 
                                height=60, 
                                disabled=True,
                                key=f"desc_{i}"
                            )
                            
                            if show_distance and hasattr(obj, 'metadata') and obj.metadata.distance is not None:
                                st.caption(f"🎯 Distance score: {obj.metadata.distance:.4f}")
                            
                            st.markdown("---")
                    
                else:
                    st.warning("🔍 No matches found. Try uploading a different image.")
            else:
                st.error("❌ Failed to generate embedding from image")
    
    elif uploaded_file:
        st.info("⏳ Please wait for model initialization...")
    else:
        st.info("👆 Upload a ring photo to start searching")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>💍 Ring Scanner - Powered by OpenCLIP and Weaviate</p>
</div>
""", unsafe_allow_html=True)

# Debug information (only show if no matches found and debugging is needed)
if st.sidebar.checkbox("🐛 Debug Mode"):
    if uploaded_file and model is not None:
        st.subheader("🔧 Debug Information")
        embedding = generate_embedding(image, model, preprocess)
        if embedding:
            st.write(f"Embedding dimensions: {len(embedding)} (truncated from 512)")
            st.write("All 10 values:", embedding)
        
        # Connection status
        if collection:
            st.write("✅ Collection accessible")
        else:
            st.write("❌ Collection not accessible")
