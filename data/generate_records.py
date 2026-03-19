import os
import random

def generate_records(num_records=3050):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    print(f"Generating {num_records} dummy records in {data_dir}...")
    
    topics = ["Artificial Intelligence", "Machine Learning", "FastAPI server", "Python applications", "Cloud infrastructure", "Vector Databases", "LangChain"]
    actions = ["drastically improves", "optimizes", "accelerates", "enhances", "transforms", "streamlines"]
    targets = ["system performance", "overall scalability", "the user experience", "data processing pipelines", "security compliance"]
    
    for i in range(num_records):
        topic = random.choice(topics)
        action = random.choice(actions)
        target = random.choice(targets)
        
        content = f"Document ID: DOC-{i:05d}\n"
        content += f"Topic: {topic}\n\n"
        content += f"This confidential document details how {topic} {action} {target}. "
        content += f"It provides a comprehensive analysis of the underlying mechanisms, benchmarks, and offers practical examples of enterprise implementation.\n"
        content += f"Furthermore, it highlights the importance of continuous monitoring and automated evaluation to systematically debug and maintain optimal results in production environments.\n"
        
        filename = f"record_{i:05d}.txt"
        file_path = os.path.join(data_dir, filename)
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
            
    print(f"Successfully generated {num_records} records.")

if __name__ == "__main__":
    generate_records()
