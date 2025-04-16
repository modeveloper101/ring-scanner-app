import open_clip
from PIL import Image
import torch
from torchvision import transforms

from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import numpy as np

import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.query import MetadataQuery

import streamlit as st

import os

weaviate_url = os.environ["WEAVIATE_URL"]
weaviate_api_key = os.environ["WEAVIATE_API_KEY"]

client = weaviate.connect_to_weaviate_cloud(
  cluster_url=weaviate_url,
  auth_credentials=Auth.api_key(weaviate_api_key),
)

collection = client.collections.get("My_Data")

# NEW CODE ADDED-1
class VideoProcessor(VideoProcessorBase):
    def recv(self, frame: av.VideoFrame):
        img = frame.to_image()  # Convert to PIL.Image
        st.session_state['captured_image'] = img
        return frame

st.title("🔍 Ring Scanner")

# Start webcam
# NEW CODE ADDED-1
webrtc_streamer(key="scanner", video_processor_factory=VideoProcessor)

if 'captured_image' in st.session_state:
  st.image(st.session_state['captured_image'], caption="Scanned Image", use_column_width=True)
    
  if st.button("🔍 Identify Ring"):
    with torch.no_grad():
      image_input = preprocess(st.session_state['captured_image']).unsqueeze(0)
      image_features = model.encode_image(image_input)
      vector = image_features[0].tolist()
      embedding = vector[:10]
      
    results = collection.query.near_vector(
      near_vector=embedding,
      limit=1,
      return_metadata=MetadataQuery(distance=True)
    )

    if results.objects:
      obj = results.objects[0]
      st.success("✅ Match found:")
      st.write("**Name:**", obj.properties["name"])
      st.write("**Description:**", obj.properties["description"])
      st.write("**Ring ID:**", obj.properties["ringID"])
    else:
      st.warning("No match found.")

# uploaded_file = st.file_uploader("Upload a ring photo", type=["jpg", "jpeg", "png"])

# if uploaded_file:
#   image = Image.open(uploaded_file).convert('RGB')
#   st.image(image, caption="Uploaded Ring", use_column_width=True)

#   model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
#   image_input = preprocess(image).unsqueeze(0)

#   # Create embedding
#   with torch.no_grad():
#     image_features = model.encode_image(image_input)
#   test_vector = image_features[0].tolist()
#   embedding = test_vector[:10]

  # client = weaviate.connect_to_weaviate_cloud(
  #   cluster_url=weaviate_url,
  #   auth_credentials=Auth.api_key(weaviate_api_key),
  # )

  # collection = client.collections.get("My_Data")


  # results = collection.query.near_vector(
  #   near_vector=embedding,
  #   limit=1,
  #   return_metadata=MetadataQuery(distance=True)
  # )

  # if results.objects:
  #   st.success("✅ Matches found:")
  #   for i, obj in enumerate(results.objects, start=1):
  #     st.subheader(f"Match #{i}")
  #     st.write("**Name:**", obj.properties["name"])
  #     st.write("**Description:**", obj.properties["description"])
  #     st.write("**Ring ID:**", obj.properties["ringID"])
  #     st.markdown("---")
  # else:
  #   st.warning("No match found.")

  # if results.objects:
  #   obj = results.objects[0]
  #   st.success("✅ Match found:")
  #   st.write("**Name:**", obj.properties["name"])
  #   st.write("**Description:**", obj.properties["description"])
  #   st.write("**Ring ID:**", obj.properties["ringID"])
  # else:
  #   st.warning("No match found.")

