# 🚀 Research Compass - Quick Start Guide

## Step 1: Launch the Application

```bash
cd "/Users/ayushchhoker/Desktop/Study/Projects/Working/Research Compass"
python launcher.py
```

The application will open at: **http://localhost:7860**

## Step 2: Configure Settings (First Time Only)

1. **Open the web interface** in your browser at http://localhost:7860
2. **Click on "Settings" tab** in the interface
3. **Configure LLM**:
   - Select "Ollama" as provider
   - Base URL: `http://localhost:11434`
   - Click "Refresh Available Models"
   - Select "llama3.2" from the dropdown
   - Click "Test Connection" - should show "Connected to Ollama"
   - Click "Save Configuration"
4. **Configure Neo4j** (Optional - can skip):
   - URI: `neo4j://127.0.0.1:7687`
   - Username: `neo4j`
   - Password: `your_password`
   - Click "Test Connection" - should show connection success
   - Click "Save Configuration"

## Step 3: Upload Your Research Papers

1. **Go to "Upload & Process" tab**
2. **Upload Files**:
   - Click "Upload Files"
   - Select your PDF files from the data/docs folder
   - Enable "Build Knowledge Graph" ✅
   - Enable "Extract Metadata" ✅
   - Click "Process All"
3. **Wait for processing** - you'll see status updates

## Step 4: Ask Questions About Your Papers

1. **Go to "Research Assistant" tab**
2. **Type a question** like: "What are the main findings from these clinical trial papers?"
3. **Enable options**:
   - ✅ Use Knowledge Graph
   - ✅ Use GNN Reasoning
   - ✅ Stream Response
   - ✅ Use Cache
4. **Click "Ask Question"**
5. **Watch the response** appear word by word

## Step 5: Explore Advanced Features

### View Citation Networks
1. **Go to "Citation Explorer" tab**
2. **Enter a paper title** from your uploaded papers
3. **Click to explore** the citation network
4. **Click on nodes** to expand connections

### Get Recommendations
1. **Go to "Recommendations" tab**
2. **Enter your interests**: "clinical trials, medical research"
3. **Click "Create/Update Profile"**
4. **Get personalized suggestions** for papers to read next

### Analyze Research Trends
1. **Go to "Temporal Analysis" tab**
2. **Enter a topic**: "clinical trial protocols"
3. **Select time window**: "yearly"
4. **Click "Analyze Topic Evolution"**

## What You'll See

🔍 **Research Assistant**: GNN-powered answers with source citations  
📊 **Knowledge Graph**: Automatic entity and relationship extraction  
💡 **Recommendations**: AI-suggested papers based on your interests  
🕸️ **Citation Networks**: Interactive exploration of research connections  
📈 **Temporal Analysis**: How research topics evolve over time  
🎨 **Visualizations**: Interactive graphs and attention maps  

## Troubleshooting

**If the application doesn't start:**
```bash
# Check if Ollama is running
ollama list

# Start Ollama
ollama serve

# Check if port 7860 is available
lsof -i :7860
```

**If you see connection errors:**
- Make sure Ollama is installed and running
- Check that the base URL is correct: http://localhost:11434
- Try refreshing the page and clicking "Test Connection" again

## Success Indicators

✅ **Application launches** at http://localhost:7860  
✅ **Settings show** "Connected to Ollama" and "Connected to Neo4j"  
✅ **Documents process** and show in the knowledge graph  
✅ **Questions generate** GNN-powered responses  
✅ **All features accessible** through the web interface  

---

**Your Research Compass is now ready for advanced research exploration!** 🚀

Just run `python launcher.py` and start exploring your research papers with GNN-powered intelligence!