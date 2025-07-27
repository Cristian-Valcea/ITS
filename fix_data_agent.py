#!/usr/bin/env python3
"""
Script to fix the corrupted DataAgent file by extracting only the class definition
"""

def fix_data_agent_file():
    """Extract and fix the DataAgent class definition"""
    
    with open('src/agents/data_agent.py', 'r') as f:
        lines = f.readlines()
    
    # Find class start and collect properly indented class content
    fixed_lines = []
    in_class = False
    skip_main_block = False
    brace_count = 0
    
    for i, line in enumerate(lines):
        # Start of class
        if line.strip().startswith('class DataAgent'):
            in_class = True
            fixed_lines.append(line)
            continue
            
        if not in_class:
            # Before class - keep imports and other module-level code
            if not line.strip().startswith('class '):
                fixed_lines.append(line)
            continue
            
        # Inside class
        if in_class:
            # Skip the misplaced main block
            if ('import logging' in line and line.startswith('    import')) or \
               ('logging.basicConfig' in line) or \
               ("config = {" in line and line.startswith('    config')) or \
               ('data_agent = DataAgent(config=config)' in line) or \
               ('print(' in line and line.startswith('    print')) or \
               ('df_bars = data_agent.run(' in line):
                skip_main_block = True
                continue
                
            # End of misplaced main block - look for real methods
            if skip_main_block and line.strip().startswith('def ') and line.startswith('    def'):
                skip_main_block = False
                fixed_lines.append(line)
                continue
                
            # Skip lines in main block
            if skip_main_block:
                continue
                
            # Keep properly indented class content
            if line.startswith('    ') or line.strip() == '':
                fixed_lines.append(line)
            elif line.strip().startswith('#') and not line.startswith('# '):
                # Module level comment after class - end class
                break
    
    # Add proper class ending and main block
    fixed_lines.append('\n')
    fixed_lines.append('# --- End of DataAgent class ---\n')
    fixed_lines.append('\n')
    fixed_lines.append('if __name__ == \'__main__\':\n')
    fixed_lines.append('    import logging\n')
    fixed_lines.append('    logging.basicConfig(level=logging.INFO, format=\'%(asctime)s - %(name)s - %(levelname)s - %(message)s\')\n')
    fixed_lines.append('\n')
    fixed_lines.append('    # Example Configuration\n')
    fixed_lines.append('    config = {\n')
    fixed_lines.append('        \'data_dir_raw\': \'data/raw_test\',\n')
    fixed_lines.append('        \'default_symbol\': \'DUMMY\'\n')
    fixed_lines.append('    }\n')
    fixed_lines.append('\n')
    fixed_lines.append('    data_agent = DataAgent(config=config)\n')
    fixed_lines.append('    print("DataAgent example run complete.")\n')
    
    # Write fixed file
    with open('src/agents/data_agent_fixed.py', 'w') as f:
        f.writelines(fixed_lines)
    
    print(f"✅ Created fixed file with {len(fixed_lines)} lines")
    
    # Test the fixed file
    try:
        import sys
        sys.path.insert(0, 'src/agents')
        
        # Import the fixed module
        import importlib.util
        spec = importlib.util.spec_from_file_location("data_agent_fixed", "src/agents/data_agent_fixed.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        DataAgent = module.DataAgent
        methods = [method for method in dir(DataAgent) if not method.startswith('_')]
        print(f"✅ Fixed DataAgent has {len(methods)} methods")
        
        required = ['get_account_summary', 'subscribe_live_bars', 'unsubscribe_live_bars']
        for method in required:
            if hasattr(DataAgent, method):
                print(f'✅ {method}')
            else:
                print(f'❌ {method}')
                
        return True
        
    except Exception as e:
        print(f"❌ Error testing fixed file: {e}")
        return False

if __name__ == "__main__":
    fix_data_agent_file()