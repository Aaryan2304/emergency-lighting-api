# 🚀 Free AI Setup Guide

## TL;DR - Recommended Setup (100% Free)

**For your RTX 3050 laptop, I recommend Google Gemini (free tier):**

```bash
# 1. Run the setup script
python setup.py

# 2. Choose option 1 (Google Gemini)
# 3. Get free API key from: https://aistudio.google.com/app/apikey
# 4. Enter the key when prompted
# 5. Done! 🎉
```

## 💰 Cost Comparison

| Backend | Cost | Quality | Setup | Local |
|---------|------|---------|-------|-------|
| **Google Gemini** | 🆓 Free tier | ⭐⭐⭐⭐ | Easy | No |
| **Ollama** | 🆓 100% Free | ⭐⭐⭐ | Medium | Yes |
| **Hugging Face** | 🆓 100% Free | ⭐⭐ | Hard | Yes |
| **OpenAI** | 💰 $0.001/call | ⭐⭐⭐⭐⭐ | Easy | No |
| **No LLM** | 🆓 Free | ⭐ | Easy | Yes |

## 🤖 Detailed Setup Options

### Option 1: Google Gemini (Recommended) 

**Why choose this:**
- Free tier with generous limits (15 requests/minute)
- High-quality AI responses
- Easy setup
- Perfect for development and testing

**Setup:**
1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Sign in with Google account
3. Click "Create API Key" 
4. Copy the key
5. Add to your `.env` file:
   ```env
   GEMINI_API_KEY=your_key_here
   GEMINI_MODEL=gemini-1.5-flash
   ```

### Option 2: Ollama (Completely Free, Local)

**Why choose this:**
- 100% free forever
- Runs on your RTX 3050 (perfect fit)
- No internet required after setup
- Complete privacy

**Setup:**
1. Download from [Ollama.ai](https://ollama.ai/)
2. Install Ollama
3. Open terminal and run:
   ```bash
   ollama pull llama3.2:3b
   ```
4. Start Ollama service:
   ```bash
   ollama serve
   ```
5. Configure in `.env`:
   ```env
   OLLAMA_BASE_URL=http://localhost:11434
   OLLAMA_MODEL=llama3.2:3b
   ```

### Option 3: Hugging Face (Local Models)

**Why choose this:**
- 100% free
- Runs locally
- Good for learning ML

**Setup:**
1. Install dependencies:
   ```bash
   pip install transformers torch accelerate
   ```
2. Configure in `.env`:
   ```env
   HF_MODEL=microsoft/DialoGPT-medium
   LOAD_HF_MODEL=true
   ```

### Option 4: No LLM (Rule-based)

**Why choose this:**
- Zero cost
- No setup required
- Still detects lighting fixtures
- Good for basic functionality

**Setup:**
Just leave LLM settings empty in `.env`

## 🔧 For Your RTX 3050 Setup

Your laptop specs are perfect for local AI:
- **RTX 3050 (4GB VRAM)**: Can run Ollama llama3.2:3b model
- **16GB RAM**: Enough for any option
- **i7-12650H**: Fast enough for real-time processing

**Recommended models for your hardware:**
- Ollama: `llama3.2:3b` (2GB model, fits perfectly)
- Hugging Face: `microsoft/DialoGPT-medium` (lighter model)

## 🚀 Quick Test

After setup, test your installation:

```bash
# Run the demo
python demo.py

# Start the API
python main.py

# Visit: http://localhost:8000/docs
```

## 💡 Pro Tips

1. **Start with Gemini** - It's the easiest and highest quality free option
2. **Try Ollama later** - For complete privacy and offline use
3. **OpenAI is optional** - Only if you need the absolute best quality
4. **The system works without LLM** - Basic detection still functions

## ❓ FAQ

**Q: Is OpenAI required?**
A: No! You have 4 free alternatives.

**Q: Which free option is best?**
A: Google Gemini for quality, Ollama for privacy.

**Q: Will this cost me money?**
A: Only if you choose OpenAI. All other options are free.

**Q: Can I change backends later?**
A: Yes! Just edit the `.env` file.

**Q: What if I don't want AI at all?**
A: The system still detects lighting fixtures using computer vision.
