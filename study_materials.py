import json
import os
from datetime import datetime
import config

class StudyMaterialsManager:
    def __init__(self):
        self.materials_file = config.STUDY_MATERIALS_FILE
        self.materials = []
        
        os.makedirs(os.path.dirname(self.materials_file), exist_ok=True)
        self.load_materials()
    
    def load_materials(self):
        """Load materials from file"""
        if os.path.exists(self.materials_file):
            try:
                with open(self.materials_file, 'r') as f:
                    self.materials = json.load(f)
            except:
                self.materials = []
        else:
            self.materials = []
    
    def save_materials(self):
        """Save materials to file"""
        try:
            with open(self.materials_file, 'w') as f:
                json.dump(self.materials, f, indent=2)
        except Exception as e:
            print(f"Save error: {e}")
    
    def add_material(self, title, content, subject="", chapter=""):
        """Add new material"""
        material = {
            'id': len(self.materials) + 1,
            'title': title,
            'content': content,
            'subject': subject,
            'chapter': chapter,
            'created_at': datetime.now().isoformat()
        }
        
        self.materials.append(material)
        self.save_materials()
        return material
    
    def get_all_materials(self):
        """Get all materials"""
        return self.materials
    
    def get_materials_for_embedding(self):
        """Get materials for embedding"""
        embedded_materials = []
        
        for material in self.materials:
            searchable_text = f"{material['title']} {material['content']}"
            
            embedded_materials.append({
                'text': searchable_text,
                'metadata': {
                    'id': material['id'],
                    'title': material['title'],
                    'subject': material['subject'],
                    'chapter': material['chapter']
                }
            })
        
        return embedded_materials
    
    def get_stats(self):
        """Get statistics"""
        subjects = {}
        for material in self.materials:
            subject = material['subject'] or 'Unknown'
            subjects[subject] = subjects.get(subject, 0) + 1
        
        return {
            'total_materials': len(self.materials),
            'subjects': subjects
        }