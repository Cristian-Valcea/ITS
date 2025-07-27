# 🔑 ITS Secrets Management - Simple Guide for Junior Developers

*"How to store and use passwords/API keys safely in our trading system"*

## 🎯 What is This?

Our trading system needs to connect to different services:
- **Interactive Brokers** (for trading)
- **Database** (for storing data) 
- **APIs** (for market data)

Instead of putting passwords directly in code (BAD!), we use a **secrets management system** that stores them safely.

## 🚀 Quick Start (5 minutes)

### 1. Install the Tool
```bash
# Just copy-paste this:
pip install cryptography argon2-cffi pydantic click PyYAML
```

### 2. Store Your First Secret
```bash
# Store an API key safely:
python cloud_secrets_cli.py set my-api-key "sk-1234567890"

# You'll be asked for a master password - remember this!
```

### 3. Get Your Secret Back
```bash
# Retrieve the API key:
python cloud_secrets_cli.py get my-api-key

# Enter the same master password
```

**🎉 Congratulations! You just stored and retrieved a secret safely.**

## 📚 Common Tasks

### Storing Different Types of Secrets

```bash
# API keys
python cloud_secrets_cli.py set openai-api-key "sk-abcdef123456"

# Database passwords  
python cloud_secrets_cli.py set db-password "super-secure-password"

# Trading credentials
python cloud_secrets_cli.py set ib-username "myusername"
python cloud_secrets_cli.py set ib-password "mypassword"
```

### Viewing What You Have Stored
```bash
# List all your secrets (names only, not values!)
python cloud_secrets_cli.py list

# Get detailed info about a secret
python cloud_secrets_cli.py info my-api-key
```

### Deleting Secrets
```bash
# Remove a secret you don't need
python cloud_secrets_cli.py delete old-api-key
```

## 💻 Using Secrets in Python Code

### Simple Example
```python
# In your trading script:
import subprocess
import json

def get_secret(secret_name):
    """Get a secret value safely"""
    result = subprocess.run([
        'python', 'cloud_secrets_cli.py', 'get', secret_name, '--json'
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        data = json.loads(result.stdout)
        return data['value']
    else:
        raise Exception(f"Failed to get secret: {secret_name}")

# Use it in your code:
api_key = get_secret("openai-api-key")
db_password = get_secret("db-password")

# Now use api_key and db_password in your connections
```

### Trading System Example
```python
# For Interactive Brokers connection:
def connect_to_ib():
    username = get_secret("ib-username")
    password = get_secret("ib-password")
    
    # Use these to connect to IB
    # (Your existing IB connection code here)
    pass

# For database connection:
def connect_to_database():
    db_password = get_secret("timescaledb-password")
    
    connection_string = f"postgresql://trading_user:{db_password}@localhost/trading_db"
    # Use connection_string to connect
    pass
```

## 🔒 Security Rules (Important!)

### ✅ DO:
- Always use the secrets system for passwords/API keys
- Use strong master passwords (ask senior dev for help)
- Keep your master password secret
- Store secrets with descriptive names like `ib-api-key`

### ❌ DON'T:
- Put passwords directly in Python files
- Share your master password with anyone
- Commit secrets to Git
- Use weak passwords like `password123`

### Example of BAD Code (Never do this!):
```python
# ❌ NEVER DO THIS:
api_key = "sk-1234567890"  # BAD - password in code!
password = "mypassword"    # BAD - visible to everyone!
```

### Example of GOOD Code:
```python
# ✅ GOOD - using secrets system:
api_key = get_secret("openai-api-key")     # GOOD!
password = get_secret("ib-password")       # GOOD!
```

## 🆘 When Things Go Wrong

### "Command not found" Error
```bash
# Make sure you're in the right directory:
cd /home/cristian/IntradayTrading/ITS

# Then try again:
python cloud_secrets_cli.py list
```

### "Secret not found" Error
```bash
# Check what secrets you have:
python cloud_secrets_cli.py list

# Maybe you misspelled the name?
```

### "Wrong password" Error
```bash
# You entered the wrong master password
# Try again with the correct password
# If you forgot it, ask a senior developer
```

### "Permission denied" Error
```bash
# Check if the secrets file exists and you can access it:
ls -la secrets.vault

# Ask a senior developer if you need help
```

## 📖 Step-by-Step Examples

### Setting Up Trading Credentials
```bash
# Step 1: Store IB credentials
python cloud_secrets_cli.py set ib-username "your_ib_username"
python cloud_secrets_cli.py set ib-password "your_ib_password"  
python cloud_secrets_cli.py set ib-api-key "your_ib_api_key"

# Step 2: Store database password
python cloud_secrets_cli.py set timescaledb-password "your_db_password"

# Step 3: Store external API keys
python cloud_secrets_cli.py set alpha-vantage-key "your_av_key"
python cloud_secrets_cli.py set yahoo-finance-key "your_yf_key"

# Step 4: Verify everything is stored
python cloud_secrets_cli.py list
```

### Using in Your Trading Script
```python
# trading_bot.py
import subprocess
import json

def get_secret(name):
    """Helper function to get secrets"""
    result = subprocess.run([
        'python', 'cloud_secrets_cli.py', 'get', name, '--json'
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        return json.loads(result.stdout)['value']
    else:
        print(f"Error getting secret {name}: {result.stderr}")
        return None

def main():
    # Get all the secrets you need
    ib_username = get_secret("ib-username")
    ib_password = get_secret("ib-password")
    db_password = get_secret("timescaledb-password")
    
    if not all([ib_username, ib_password, db_password]):
        print("Missing required secrets!")
        return
    
    print("All secrets loaded successfully!")
    
    # Now use them in your trading logic
    # connect_to_ib(ib_username, ib_password)
    # connect_to_database(db_password)
    # run_trading_strategy()

if __name__ == "__main__":
    main()
```

## 🎓 Advanced Features (When You're Ready)

### Using Different Environments
```bash
# For development
python cloud_secrets_cli.py set dev-api-key "development-key"

# For production (when you're more experienced)
python cloud_secrets_cli.py --backend aws set prod-api-key "production-key"
```

### Adding Descriptions
```bash
# Add helpful descriptions to your secrets
python cloud_secrets_cli.py set ib-api-key "your-key" --description "Interactive Brokers API key for paper trading"
```

### Checking Secret Information
```bash
# See when a secret was created/updated
python cloud_secrets_cli.py info ib-api-key
```

## 📞 Getting Help

### Command Help
```bash
# See all available commands
python cloud_secrets_cli.py --help

# Get help for a specific command
python cloud_secrets_cli.py set --help
```

### Common Commands Cheat Sheet
```bash
# Store a secret
python cloud_secrets_cli.py set <name> "<value>"

# Get a secret
python cloud_secrets_cli.py get <name>

# List all secrets
python cloud_secrets_cli.py list

# Delete a secret
python cloud_secrets_cli.py delete <name>

# Get info about a secret
python cloud_secrets_cli.py info <name>
```

### When to Ask for Help
- You forgot your master password
- You get permission errors
- You need to set up cloud backends (AWS, Azure)
- You're not sure if you're doing something securely
- Any error you can't understand

### Who to Ask
1. **Senior developers** on the team
2. **System administrator** for permission issues
3. **Security team** for password policy questions

## 🎯 Success Checklist

### ✅ You're Ready When You Can:
- [ ] Store a secret using the CLI
- [ ] Retrieve a secret using the CLI
- [ ] List your stored secrets
- [ ] Use secrets in Python code with `get_secret()` function
- [ ] Know what NOT to put in Git (passwords!)
- [ ] Ask for help when you need it

### 🏆 Next Level Skills:
- [ ] Understand different backends (local vs cloud)
- [ ] Use environment-specific configurations
- [ ] Set up monitoring and health checks
- [ ] Help other junior developers

## 🎉 You're Done!

You now know how to:
- Store passwords and API keys safely
- Use them in your Python code
- Follow security best practices
- Get help when needed

**Remember**: When in doubt, ask a senior developer. It's better to ask questions than to accidentally expose passwords!

---

*This guide covers the basics. For advanced features, see the complete documentation in `/documents/126_PHASE3_FINAL_STATUS.md`*