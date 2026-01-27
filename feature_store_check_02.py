import hopsworks
import os
from dotenv import load_dotenv

load_dotenv()

project = hopsworks.login(
    project=os.getenv("HOPSWORKS_PROJECT"),
    api_key_value=os.getenv("HOPSWORKS_API_KEY")
)

fs = project.get_feature_store()
print("✅ Connected to Feature Store:", fs.name)
