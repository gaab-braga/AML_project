import re

# Read the notebook file
with open('c:/Users/gafeb/OneDrive/Desktop/lavagem_dev/notebooks/02_FeatureEngineering_e_Pipelines.ipynb', 'r', encoding='utf-8') as f:
    content = f.read()

# Remove emojis
emoji_pattern = r'[🧹🕸️🔧✅❌🚀📡⚠️🌐⏳]'
content = re.sub(emoji_pattern, '', content)

# Write back
with open('c:/Users/gafeb/OneDrive/Desktop/lavagem_dev/notebooks/02_FeatureEngineering_e_Pipelines.ipynb', 'w', encoding='utf-8') as f:
    f.write(content)

print('Emojis removidos com sucesso!')