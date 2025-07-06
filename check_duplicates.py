#!/usr/bin/env python3
"""
Check for duplicate EventType/EventPriority/RiskEvent declarations
that could break isinstance() checks.
"""

import os
import re
from pathlib import Path

def find_class_definitions(directory):
    """Find all class definitions in Python files."""
    definitions = {}
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                filepath = Path(root) / file
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                    # Look for class definitions
                    patterns = [
                        r'class\s+(EventType)\s*\(',
                        r'class\s+(EventPriority)\s*\(',
                        r'class\s+(RiskEvent)\s*[:\(]',
                    ]
                    
                    for pattern in patterns:
                        matches = re.finditer(pattern, content, re.MULTILINE)
                        for match in matches:
                            class_name = match.group(1)
                            line_num = content[:match.start()].count('\n') + 1
                            
                            if class_name not in definitions:
                                definitions[class_name] = []
                            
                            definitions[class_name].append({
                                'file': str(filepath),
                                'line': line_num,
                                'match': match.group(0)
                            })
                            
                except Exception as e:
                    print(f"Error reading {filepath}: {e}")
    
    return definitions

def main():
    print("🔍 CHECKING FOR DUPLICATE CLASS DEFINITIONS")
    print("=" * 60)
    
    risk_dir = "src/risk"
    definitions = find_class_definitions(risk_dir)
    
    duplicates_found = False
    
    for class_name, locations in definitions.items():
        print(f"\n📋 {class_name} definitions found: {len(locations)}")
        
        if len(locations) > 1:
            print(f"⚠️  DUPLICATE DEFINITIONS DETECTED for {class_name}!")
            duplicates_found = True
        
        for i, loc in enumerate(locations, 1):
            status = "✅" if len(locations) == 1 else "❌"
            print(f"   {status} {i}. {loc['file']}:{loc['line']} - {loc['match']}")
    
    print(f"\n{'='*60}")
    if duplicates_found:
        print("❌ DUPLICATE DEFINITIONS FOUND!")
        print("   This will break isinstance() checks.")
        print("   Fix: Remove duplicate definitions and import from event_types.py")
    else:
        print("✅ NO DUPLICATE DEFINITIONS FOUND")
        print("   All classes are properly defined once and imported elsewhere.")
    
    return duplicates_found

if __name__ == "__main__":
    has_duplicates = main()
    if has_duplicates:
        exit(1)